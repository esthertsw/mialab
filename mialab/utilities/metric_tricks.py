import os
import numpy as np
import pandas as pd
import math
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pymia.evaluation.metric as pymia_metrics

import pymia.evaluation.evaluator as pymia_evaluator



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


def mask_dilation_and_erosion(seg: sitk.Image, gt = sitk.Image, result_dir=None, img_id=None):
    """Dilate segmentations per label specified

    Args:
        seg (sitk.Image): Segmented image
        result_dir (str): Storage directory for image after manipulation of labels
        img_id (str): For identifying the stored images

    Returns:
        sitk.Image: Segmented image with adjusted labels
    """    
    seg_np = to_np(seg)
    gt_np = to_np(gt)
    out = np.zeros_like(seg_np, dtype=np.uint8)
    labels = range(1,6)
    max_voxel_diff = 0
    max_voxels_label = 0
    for label in labels:
        mask = (seg_np == label).astype(np.int8)
        gt_mask = (gt_np == label).astype(np.int8)
        voxel_diff = np.sum(gt_mask) - np.sum(mask)
        if abs(voxel_diff) > abs(max_voxel_diff):
            max_voxels_label = label
            max_voxel_diff = voxel_diff
        out[mask == 1] = label

    # Apply morphological changes on the label with max voxel diff
    mask = (seg_np == max_voxels_label).astype(np.int8)
    mask_img = sitk.GetImageFromArray(mask)
    mask_img.CopyInformation(seg)

    # GT is larger than pred for that label's volume --> Apply dilation
    changed_voxels = 0
    if max_voxel_diff > 0:
        # Dilate and erode feature labels
        mask_img = sitk.BinaryMorphologicalClosing(mask_img, [2,2,1])
        # Overwrite the new region's labels
        mask = to_np(mask_img)
        out[out == max_voxels_label] = 0
        out[mask == 1] = max_voxels_label
        changed_voxels = np.sum((gt_np == max_voxels_label).astype(np.int8)) - np.sum(mask)
        print("Closing")
    else:
        # Erode and dilate feature labels
        mask_img = sitk.BinaryMorphologicalOpening(mask_img, [2,2,1])
        # Overwrite the new region's labels
        mask = to_np(mask_img)
        out[out == max_voxels_label] = 0
        out[mask == 1] = max_voxels_label
        changed_voxels = np.sum((gt_np == max_voxels_label).astype(np.int8)) - np.sum(mask)
        print("Opening")
    print("Applied volume morph changes to ", max_voxels_label)
    print("Diff before: ", max_voxel_diff)
    print("Diff after: ", changed_voxels)
    out_img = sitk.GetImageFromArray(out.astype(np.uint8))
    out_img.CopyInformation(seg)
    if result_dir and img_id:
        sitk.WriteImage(out_img, os.path.join(result_dir, img_id + '_TRICKED_volumeMorph.mha'), True)
    return out_img

def relabel(seg: sitk.Image, gt: sitk.Image):
    if seg is None or gt is None:
        print("relabel(): missing arguments")
        return
    print("Running relabel(), args received of type ", type(seg), type(gt))
    std_to_new_label_mapping = { # original label names kept for visibility
            1: [1, "WhiteMatter"],
            2: [2, "GreyMatter"],
            3: [6, "Hippocampus"],
            4: [6, "Amygdala"],
            5: [6,"Thalamus"],
        } # where 6 = "SmallStructures"
     
    imgs = [seg, gt]
    for idx, img in enumerate(imgs):
        img_as_np = to_np(img)
        out = np.zeros_like(img_as_np, dtype=np.uint8)
        for label, mapping in std_to_new_label_mapping.items():
            # Get a mask of points where label == label in loop
            mask = (img_as_np == label).astype(np.uint8)
            # Reassign with new label
            out[mask == 1] = mapping[0]
        out_img = sitk.GetImageFromArray(out.astype(np.uint8))
        out_img.CopyInformation(img)
        imgs[idx] = out_img
    return imgs


# Metric evaluation
def get_metrics():
    """Return the list of metrics used for the trick experiments."""
    return [
        pymia_metrics.DiceCoefficient(),
        pymia_metrics.JaccardCoefficient(),
        pymia_metrics.VolumeSimilarity(),
        pymia_metrics.AverageDistance(),
        pymia_metrics.HausdorffDistance(percentile=95, metric="HDRFDST"),
    ]


def evaluate(pred: sitk.Image, gt: sitk.Image, relabeled=False):
    """
    Evaluate a prediction against ground truth using the same style
    of metrics/labels as the main pipeline.
    """
    # Label mapping consistent with pipeline_utilities.init_evaluator()
    if relabeled: # To allow for the assessment of labels with changed granularity
        labels = {
            1: "WhiteMatter",
            2: "GreyMatter",
            6: "SmallStructures"
        }
    else:
        labels = {
            1: "WhiteMatter",
            2: "GreyMatter",
            3: "Hippocampus",
            4: "Amygdala",
            5: "Thalamus",
        }

    evaluator = pymia_evaluator.SegmentationEvaluator(get_metrics(), labels)
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
    needs_gt_relabel=False,
    needs_gt_morph=False
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
    new_labels = {1: 'WhiteMatter', 2: 'GreyMatter', 6: 'SmallStructures'}
    # Apply trick
    if needs_gt_relabel: # Special case if relabeling is done, GT needs to be relabeled too
        manipulated, gt_relabeled = trick_fn(seg,gt)
    elif needs_gt_morph:
        manipulated = trick_fn(seg,gt)
        gt_relabeled = gt
    else: 
        manipulated = trick_fn(seg)
        gt_relabeled = gt

    # Save images
    save_slice(seg, os.path.join(outdir, f"{name}_orig.png"))
    save_slice(manipulated, os.path.join(outdir, f"{name}_tricked.png"))

    # Evaluate original vs manipulated
    orig_results = evaluate(seg, gt)
    trick_results = evaluate(manipulated, gt_relabeled, relabeled=needs_gt_relabel)

    if not needs_gt_relabel:
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
    else:
        relabeled_csv_path = os.path.join(outdir, f"{name}_metrics.csv")
        with open(relabeled_csv_path, "w") as f:
            f.write("Label,Metric,Value\n")
            # Store all original labels and their results
            for r in orig_results:
                f.write(f"{r.label},{r.metric},{r.value}\n")
            # Store only results from the SmallStructures label
            for x in trick_results:
                if x.label == 'SmallStructures':
                    f.write(f"{x.label},{x.metric},{x.value}\n")

    if needs_gt_relabel:
        return manipulated, gt_relabeled
    else:
        return manipulated


def load_trick_results(tricks_dir: str, relabeled=False) -> pd.DataFrame:
    """
    Load all *_metrics.csv produced by run_metric_trick_experiment.
    Returns a tidy DataFrame with columns:
        Subject, Label, Metric, Original, Tricked, Trick
    """
    rows = []
    identifier = "_metrics.csv" if not relabeled else "_relabeled_metrics.csv"

    for fname in os.listdir(tricks_dir):
        if fname.endswith(identifier):
            trick_name = fname.replace('_metrics.csv', "")
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
    if not len(tricks) or not len(tricks_df['Original']) or not len(tricks_df['Tricked']):
        print("Empty column in csv, unable to plot summary boxplots.")
        return
    
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

def plot_relabeled_summary_boxplots(tricks_df: pd.DataFrame, outdir: str):
    metrics = tricks_df["Metric"].unique() # N metrics used

    if not len(tricks_df['Value']):
        print("Empty column in csv, unable to plot relabeled images\' boxplots.")
        return
    
    # Create side-by-side boxplot charts assessing label structures, for all metrics 
    fig, axs = plt.subplots(nrows=len(metrics), ncols=2, sharey=False, figsize=(6,15))
    for i, metric in enumerate(metrics):
        # Original plot on the left
        orig_df = tricks_df[(tricks_df["Metric"] == metric) & (tricks_df["Label"] != "SmallStructures")]
        orig_labels = orig_df["Label"].unique()
        orig_box_data = [
            orig_df.loc[orig_df["Label"] == lbl, "Value"].values
            for lbl in orig_labels
        ]
        y_min = 0
        y_max = math.ceil(np.concatenate(orig_box_data).max())

        axs[i, 0].boxplot(
                        orig_box_data,
                        labels=orig_labels,
                        showfliers=False
                    )
        axs[i, 0].set_ylim(y_min, y_max)
        axs[i, 0].set_ylabel(metric, fontsize=10)
        axs[i, 0].grid(alpha=0.3)

        # New labels on the right
        mapping = {
            "Hippocampus": "SmallStructures",
            "Amygdala": "SmallStructures",
            "Thalamus": "SmallStructures"
        }
        sub = tricks_df[tricks_df["Metric"] == metric].copy()
        sub["Relabeled"] = sub["Label"].map(mapping).fillna(sub["Label"])

        rel_labels = sub["Relabeled"].unique()

        rel_box_data = [
            sub.loc[sub["Relabeled"] == lbl, "Value"].values
            for lbl in rel_labels
        ]
        axs[i, 1].boxplot(
                        rel_box_data,
                        labels=rel_labels,
                        showfliers=False
                    )
        axs[i, 1].grid(alpha=0.3)

        # Set y axis limits
        axs[i, 0].set_ylim(y_min, y_max)
        axs[i, 1].set_ylim(y_min, y_max)
        # Set x label size
        axs[i, 0].tick_params(axis='x', labelsize=8)
        axs[i, 1].tick_params(axis='x', labelsize=8)
        # Rotate all x labels
        for ax in axs[i]:
            ax.tick_params(axis="x", labelrotation=30, labelsize=8)

    axs[0,0].set_title('Original labeling', fontsize=10)
    axs[0,1].set_title('Grouped labeling', fontsize=10)
    

    plt.suptitle(f"Metric Vulnerability Experiment — Label Granularity")
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    fig.savefig(os.path.join(outdir, f"relabeled_FULL_summary_boxplot.png"), dpi=150)
    plt.close(fig)
