import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pymia.evaluation.evaluator import Evaluator
from pymia.evaluation.metric import DiceCoefficient, HausdorffDistance, AverageDistance, JaccardCoefficient, VolumeSimilarity


from pymia.evaluation.evaluator import SegmentationEvaluator
from pymia.evaluation.metric import (
    DiceCoefficient,
    HausdorffDistance,
    AverageDistance,
    JaccardCoefficient,
    VolumeSimilarity,
)



def to_np(img: sitk.Image) -> np.ndarray:
    """Convert a SimpleITK image to a NumPy array (z, y, x)."""
    return sitk.GetArrayFromImage(img)


# Manipulation Tricks
def keep_largest_cc(seg: sitk.Image) -> sitk.Image:
    """
    Keep only the largest connected component per label.
    This tends to remove small scattered predictions far from the main blob.
    """
    seg_np = to_np(seg)
    out = np.zeros_like(seg_np, dtype=np.uint8)

    unique_labels = np.unique(seg_np)
    for label in unique_labels:
        if label == 0:
            continue

        mask = (seg_np == label).astype(np.uint8)
        mask_img = sitk.GetImageFromArray(mask)
        mask_img.CopyInformation(seg)

        cc = sitk.ConnectedComponent(mask_img)
        # Relabel: largest component gets label 1, others 2, 3, ...
        largest = sitk.RelabelComponent(cc, sortByObjectSize=True)
        largest_np = sitk.GetArrayFromImage(largest)

        # keep only the largest CC for this label
        out[largest_np == 1] = label

    out_img = sitk.GetImageFromArray(out.astype(np.uint8))
    out_img.CopyInformation(seg)
    return out_img


def shrink_boundary(seg: sitk.Image, radius: float = 0.5) -> sitk.Image:
    """
    Shrink the segmentation inward by removing a shell at the boundary.

    This is done via the distance map:
    - Compute distance from the boundary
    - Keep only voxels whose distance >= radius
    """
    seg_bin = seg > 0
    # insideIsPositive=True → inside > 0 is positive distance
    dist = sitk.SignedMaurerDistanceMap(seg_bin, insideIsPositive=True, squaredDistance=False)
    shrunk_mask = sitk.Cast(dist > radius, sitk.sitkUInt8)
    shrunk = sitk.Mask(seg, shrunk_mask)
    return shrunk


def remove_far_voxels(seg: sitk.Image,
                      frac_per_label: dict | None = None,
                      default_frac: float = 0.7) -> sitk.Image:
    """
    For each anatomical label, remove voxels that are 'far' from the label-specific centroid.

    - We compute distances within each label separately.
    - For label L, we keep only voxels whose distance <= frac * max_distance_L,
      where frac is taken from `frac_per_label` or `default_frac`.

    This simulates an over-compact, 'too central' segmentation while
    being less insane than one global radius.
    """

    # default fractions per label (adapt/tune if you want)
    # labels: 1=WM, 2=GM, 3=Hippocampus, 4=Amygdala, 5=Thalamus

    if frac_per_label is None:
        frac_per_label = {
            1: 0.95,  # White matter: keep 95% of radial extent
            2: 0.95,  # Grey matter: same
            3: 0.70,  # Hippocampus: keep inner 70%
            4: 0.70,  # Amygdala
            5: 0.80,  # Thalamus
        }

    seg_np = to_np(seg)
    out = np.zeros_like(seg_np, dtype=np.uint8)

    labels = [l for l in np.unique(seg_np) if l != 0]

    for lbl in labels:
        coords = np.column_stack(np.where(seg_np == lbl))
        if coords.size == 0:
            continue

        # centroid of THIS label
        center = coords.mean(axis=0)

        # distances of each voxel of this label to its centroid
        dists = np.linalg.norm(coords - center, axis=1)
        max_d = dists.max()

        frac = frac_per_label.get(lbl, default_frac)
        thr = frac * max_d  # label-specific threshold

        keep_mask = dists <= thr
        kept_coords = coords[keep_mask]

        for c in kept_coords:
            out[tuple(c)] = lbl

    new_img = sitk.GetImageFromArray(out.astype(np.uint8))
    new_img.CopyInformation(seg)
    return new_img


def mask_dilation_and_erosion(seg: sitk.Image, result_dir=None, img_id=None):
    """Dilate segmentations per label specified

    Args:
        seg (sitk.Image): Segmented image
        result_dir (str): Storage directory for image after manipulation of labels
        img_id (str): For identifying the stored images

    Returns:
        sitk.Image: Segmented image with adjusted labels
    """    
    seg_np = to_np(seg)
    out = np.zeros_like(seg_np, dtype=np.uint8)
    labels_descending_size = [1,2,5,3,4] # Descending sizes: white matter → gray matter → thalamus → hippocampus → amygdala
    for label in labels_descending_size:
        mask = (seg_np == label).astype(np.uint8)
        mask_img = sitk.GetImageFromArray(mask)
        mask_img.CopyInformation(seg)

        # Dilate and erode feature labels
        mask_img = sitk.BinaryMorphologicalClosing(mask_img, [3,3,3])

        # Select largest connected component from the closed mask
        cc = sitk.ConnectedComponent(mask_img)
        # Relabel: largest component gets label 1, others 2, 3, ...
        largest = sitk.RelabelComponent(cc, sortByObjectSize=True)
        largest_np = sitk.GetArrayFromImage(largest)

        # keep only the largest CC for this label
        # NOTE: later labels processed will overwrite previous assignments if any. Therefore larger segments should be processed first
        out[largest_np == 1] = label

    out_img = sitk.GetImageFromArray(out.astype(np.uint8))
    out_img.CopyInformation(seg)
    # if result_dir and img_id:
        # sitk.WriteImage(out_img, os.path.join(result_dir, img_id + '_TRICKED_morph_closed.mha'), True)
    return out_img

def relabel(seg: sitk.Image, gt: sitk.Image):
    std_to_new_label_mapping = { # original label names kept for visibility
            3: [6, "Hippocampus"],
            4: [6, "Amygdala"],
            5: [6,"Thalamus"],
        } # where 6 = "SmallStructures"
     
    imgs = [seg, gt]
    for idx, img in enumerate(imgs):
        img_as_np = to_np(img)
        for label, mapping in std_to_new_label_mapping.items():
            mask = (img_as_np == label).astype(np.uint8)
            img_as_np[mask] = mapping[0] # re-assign to new label value
        imgs[idx] = (sitk.GetImageFromArray(img_as_np)).CopyInformation(img)
    return imgs


# Metric evaluation
def get_metrics():
    """Return the list of metrics used for the trick experiments."""
    return [
        DiceCoefficient(),
        JaccardCoefficient(),
        VolumeSimilarity(),
        AverageDistance(),
        HausdorffDistance(percentile=95, metric="HDRFDST"),
    ]


def evaluate(pred: sitk.Image, gt: sitk.Image, relabeled=None):
    """
    Evaluate a prediction against ground truth using the same style
    of metrics/labels as the main pipeline.
    """
    # Label mapping consistent with pipeline_utilities.init_evaluator()
    if relabeled: # To allow for the assessment of labels with changed granularity
        labels = relabeled
    else:
        labels = {
            1: "WhiteMatter",
            2: "GreyMatter",
            3: "Hippocampus",
            4: "Amygdala",
            5: "Thalamus",
        }

    evaluator = SegmentationEvaluator(get_metrics(), labels)
    evaluator.evaluate(pred, gt, "exp")
    return evaluator.results


# Save slice for visualization
def save_slice(img: sitk.Image, out_path: str, cmap: str = "tab20"):
    """
    Save a mid-axial slice of a 3D image for quick visual inspection.
    For labels, a qualitative colormap like 'tab20' is nice.
    """
    arr = to_np(img)
    mid = arr.shape[0] // 2
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure(figsize=(5, 5))
    plt.imshow(arr[mid], cmap=cmap)
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close()



    
# Main experiment function
def run_metric_trick_experiment(
    seg: sitk.Image,
    gt: sitk.Image,
    outdir: str,
    name: str,
    trick_fn,
):
    """
    Run one manipulation experiment:

    - apply trick_fn(seg) to get a manipulated segmentation
    - save mid-slices of original vs manipulated
    - compute metrics (Dice, Jaccard, VS, AvgDist, HD95) for both
    - dump a CSV comparing original vs tricked per label/metric

    Returns:
        manipulated (sitk.Image): the manipulated segmentation
    """
    os.makedirs(outdir, exist_ok=True)

    # Apply trick
    if trick_fn == relabel: # Special case if relabeling is done, GT needs to be relabeled too
        manipulated, gt_relabeled = trick_fn(seg,gt)
    manipulated = trick_fn(seg)

    # Save images
    save_slice(seg, os.path.join(outdir, f"{name}_orig.png"))
    save_slice(manipulated, os.path.join(outdir, f"{name}_tricked.png"))

    # Evaluate original vs manipulated
    orig_results = evaluate(seg, gt)
    trick_results = evaluate(manipulated, gt_relabeled if trick_fn == relabel else gt, relabeled=True if trick_fn == relabel else False)

    csv_path = os.path.join(outdir, f"{name}_metrics.csv")
    with open(csv_path, "w") as f:
        f.write("Label,Metric,Original,Tricked\n")
        for r in orig_results:
            # r has attributes: id_, label, metric, value
            matches = [
                x for x in trick_results
                if x.label == r.label and x.metric == r.metric
            ]
            if matches:
                f.write(f"{r.label},{r.metric},{r.value},{matches[0].value}\n")
                



    return manipulated if trick_fn != relabel else manipulated, gt_relabeled

import pandas as pd
import os

def load_trick_results(tricks_dir: str) -> pd.DataFrame:
    """
    Load all *_metrics.csv produced by run_metric_trick_experiment.
    Returns a tidy DataFrame with columns:
        Subject, Label, Metric, Original, Tricked, Trick
    """
    rows = []

    for fname in os.listdir(tricks_dir):
        if fname.endswith("_metrics.csv"):
            trick_name = fname.replace("_metrics.csv", "")
            subject = trick_name.split("_")[0]
            trick = trick_name.split("_", 1)[1]

            df = pd.read_csv(os.path.join(tricks_dir, fname))

            df["Subject"] = subject
            df["Trick"] = trick

            rows.append(df)

    return pd.concat(rows, ignore_index=True)


def plot_trick_summary_boxplot(tricks_df: pd.DataFrame, outdir: str):
    """
    Creates ONE figure per trick.
    Each figure contains:
        - rows = structures (labels)
        - columns = metrics
        - each cell = boxplot of Original vs Tricked across subjects
    """
    labels = tricks_df["Label"].unique()
    metrics = tricks_df["Metric"].unique()
    tricks = tricks_df["Trick"].unique()

    # figure size: adapt to number of metrics/labels
    n_rows = len(labels)
    n_cols = len(metrics)

    for trick in tricks:
        df_t = tricks_df[tricks_df["Trick"] == trick]

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(3*n_cols, 2.5*n_rows),
            squeeze=False
        )

        for i, label in enumerate(labels):
            df_l = df_t[df_t["Label"] == label]

            for j, metric in enumerate(metrics):
                df_m = df_l[df_l["Metric"] == metric]

                ax = axes[i][j]

                if df_m.empty:
                    ax.set_axis_off()
                    continue

                # Boxplot: original vs tricked
                ax.boxplot(
                    [df_m["Original"], df_m["Tricked"]],
                    labels=["Orig", "Trick"],
                    showfliers=False
                )

                if i == 0:
                    ax.set_title(metric, fontsize=10)

                if j == 0:
                    ax.set_ylabel(label, fontsize=10)

                ax.grid(alpha=0.3)

        plt.suptitle(f"Metric Vulnerability Summary — Trick: {trick}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        fig.savefig(os.path.join(outdir, f"{trick}_FULL_summary_boxplot.png"), dpi=150)
        plt.close(fig)