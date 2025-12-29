import os
import sys
import numpy as np
import pandas as pd
import math
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pymia.evaluation.metric import (
    DiceCoefficient,
    JaccardCoefficient,
    AverageDistance,
    VolumeSimilarity,
    HausdorffDistance,
)
from dataclasses import dataclass
from typing import Callable
try:
    import mialab.utilities.pipeline_utilities as putil
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[0]), '..'))
    import mialab.utilities.pipeline_utilities as putil

# ----------------
# Helper functions
# ----------------
def save_slice(img: sitk.Image, out_path: str, cmap="tab20"):
    """Save mid-axial slice of a SimpleITK image."""
    arr = sitk.GetArrayFromImage(img)
    mid = arr.shape[0] // 2
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.imshow(arr[mid], cmap=cmap)
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight", dpi=130)
    plt.close()


def get_metrics():
    return [
        DiceCoefficient(),
        JaccardCoefficient(),
        VolumeSimilarity(),
        AverageDistance(),
        HausdorffDistance(percentile=95, metric="HDRFDST"),
    ]

def evaluate(pred: sitk.Image, gt: sitk.Image, expt_identifier='exp', relabeled=False):
    """
    Evaluate a segmentation prediction against ground truth.
    Args:
        pred (sitk.Image): image with predicted segments.
        gt (sitk.Image): image with ground truth segments.
        expt_identifier (str): name for the experiment results file.
        relabeled (bool, optional): Dictates which set of labels to use, reduced granularity or not. Defaults to False.

    Returns:
        list: list of pymia.evaluation.evaluator.Result objects for the associated metrics
    """    
    if relabeled: # To allow for the assessment of labels with reduced granularity
        labels = {
        1:"WhiteMatter", 
        2:"GreyMatter", 
        6:"SmallStructures"
    }
    else:
        labels = {
            1: "WhiteMatter",
            2: "GreyMatter",
            3: "Hippocampus",
            4: "Amygdala",
            5: "Thalamus",
        }

    evaluator = putil.init_evaluator(labels=labels, metrics=[
        DiceCoefficient(),
        JaccardCoefficient(),
        VolumeSimilarity(),
        AverageDistance(),
        HausdorffDistance(percentile=95, metric="HDRFDST"),
        ]
    )
    evaluator.evaluate(pred, gt, expt_identifier)
    return evaluator.results


# -----------------------------
# Metric Manipulation Functions
# -----------------------------

def keep_largest_cc(seg: sitk.Image) -> sitk.Image:
    """
    Keep Largest Connected Component
    For each label, keep only the largest connected component.
    
    Returns:
        sitk.Image: image made up of largest connected component per segment 
    """    
    seg_np = sitk.GetArrayFromImage(seg)
    out = np.zeros_like(seg_np, dtype=np.uint8)

    for label in np.unique(seg_np):
        if label == 0:
            continue
        mask = (seg_np == label).astype(np.uint8)
        mask_img = sitk.GetImageFromArray(mask)
        mask_img.CopyInformation(seg)

        cc = sitk.ConnectedComponent(mask_img)
        relabeled = sitk.RelabelComponent(cc, sortByObjectSize=True)
        arr = sitk.GetArrayFromImage(relabeled)

        out[arr == 1] = label

    out_img = sitk.GetImageFromArray(out)
    out_img.CopyInformation(seg)
    return out_img

def shrink_boundary(seg: sitk.Image, radius_per_label=None, default_radius=1) -> sitk.Image:
    """
    Per-label inward erosion, using BinaryErode with a small radius (in voxels)
    
    Returns:
        sitk.Image: image after boundary around each segment is reduced
    """    
    if radius_per_label is None:
        radius_per_label = {
            1: 1, 
            2: 0, 
            3: 1, 
            4: 1, 
            5: 1
        }

    seg_np = sitk.GetArrayFromImage(seg)
    out = np.zeros_like(seg_np, dtype=np.uint8)

    for lbl in np.unique(seg_np):
        if lbl == 0:
            continue
        mask_np = (seg_np == lbl).astype(np.uint8)
        mask_img = sitk.GetImageFromArray(mask_np)
        mask_img.CopyInformation(seg)

        r = radius_per_label.get(lbl, default_radius)

        if r <= 0:
            out[mask_np == 1] = lbl
            continue

        radius_vec = [int(round(r))] * seg.GetDimension()
        eroded = sitk.BinaryErode(mask_img, radius_vec, sitk.sitkBall, 1)
        eroded_np = sitk.GetArrayFromImage(eroded)
        out[eroded_np == 1] = lbl

    out_img = sitk.GetImageFromArray(out.astype(np.uint8))
    out_img.CopyInformation(seg)
    return out_img

def remove_far_voxels(seg: sitk.Image,
                      max_dist_per_label=None,
                      min_size_per_label=None,
                      default_max_dist=20,
                      default_min_size=20) -> sitk.Image:
    """
    Remove voxels components far from the largest connected component, using label-dependent distance and size thresholds.
    """
    if max_dist_per_label is None:
        max_dist_per_label = {
                                1: 30, 
                                2: 30, 
                                3: 25, 
                                4: 25, 
                                5: 20
                                }
    if min_size_per_label is None:
        min_size_per_label = {
                                1: 300, 
                                2: 300, 
                                3: 15, 
                                4: 15, 
                                5: 40
                                }
    
    seg_np = sitk.GetArrayFromImage(seg).astype(np.uint8)
    out = np.zeros_like(seg_np, dtype=np.uint8)

    for lbl in np.unique(seg_np):
        if lbl == 0:
            continue
        mask = (seg_np == lbl).astype(np.uint8)
        mask_img = sitk.GetImageFromArray(mask)
        mask_img.CopyInformation(seg)

        cc = sitk.ConnectedComponent(mask_img)
        rel = sitk.RelabelComponent(cc, sortByObjectSize=True)
        rel_np = sitk.GetArrayFromImage(rel)

        if rel_np.max() == 0:
            continue

        main = (rel_np == 1)
        main_coords = np.column_stack(np.where(main))
        main_center = main_coords.mean(axis=0)
        out[main] = lbl

        max_dist = max_dist_per_label.get(lbl, default_max_dist)
        min_size = min_size_per_label.get(lbl, default_min_size)

        for comp_id in range(2, int(rel_np.max()) + 1):
            comp = (rel_np == comp_id)
            if comp.sum() < min_size:
                continue
            comp_center = np.column_stack(np.where(comp)).mean(axis=0)
            if np.linalg.norm(comp_center - main_center) <= max_dist:
                out[comp] = lbl

    out_img = sitk.GetImageFromArray(out)
    out_img.CopyInformation(seg)
    return out_img

def mask_dilation_and_erosion(seg: sitk.Image, gt = sitk.Image, result_dir=None, img_id=None) -> sitk.Image:
    """
    Increase/decrease segmentation mask volume for the brain structure with largest volume difference between predicted and ground truth segments

    Args:
        seg (sitk.Image): Segmented image
        gt (sitk.Image): Ground truth segmented image
        result_dir (str|None): Storage directory for image after manipulation of labels
        img_id (str|None): For identifying the stored images

    Returns:
        sitk.Image: Segmented image with adjusted labels
    """    
    seg_np = sitk.GetArrayFromImage(seg)
    gt_np = sitk.GetArrayFromImage(gt)
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

    # GT is larger than pred for that label's volume --> Apply closing
    changed_voxels = 0
    if max_voxel_diff > 0:
        # Dilate and erode feature labels
        mask_img = sitk.BinaryMorphologicalClosing(mask_img, [2,2,1])
        # Overwrite the new region's labels
        mask = sitk.GetArrayFromImage(mask_img)
        out[out == max_voxels_label] = 0
        out[mask == 1] = max_voxels_label
        changed_voxels = np.sum((gt_np == max_voxels_label).astype(np.int8)) - np.sum(mask)
    else:
        # GT is smaller than pred --> Apply opening
        mask_img = sitk.BinaryMorphologicalOpening(mask_img, [2,2,1])
        # Overwrite the new region's labels
        mask = sitk.GetArrayFromImage(mask_img)
        out[out == max_voxels_label] = 0
        out[mask == 1] = max_voxels_label
        changed_voxels = np.sum((gt_np == max_voxels_label).astype(np.int8)) - np.sum(mask)
    print("Applied volume morph changes to ", max_voxels_label)
    print("Difference in voxels before: ", max_voxel_diff)
    print("Difference in voxels after: ", changed_voxels)
    out_img = sitk.GetImageFromArray(out.astype(np.uint8))
    out_img.CopyInformation(seg)
    if result_dir and img_id:
        sitk.WriteImage(out_img, os.path.join(result_dir, img_id + '_TRICKED_volumeMorph.mha'), True)
    return out_img

def relabel(seg: sitk.Image, gt: sitk.Image) -> sitk.Image:
    """
    Relabel images with a reduced-granularity label set, where SmallStructures covers the Hippocampus, Amygdala and Thalamus

    Args:
        seg (sitk.Image): Segmented image
        gt (sitk.Image): Ground truth segmented image

    Returns:
        sitk.Image: Segmented image with re-mapped labels
    """    
    if seg is None or gt is None:
        print("relabel(): missing arguments")
        return
    std_to_new_label_mapping = { # original label names kept for visibility
            1: [1, "WhiteMatter"],
            2: [2, "GreyMatter"],
            3: [6, "Hippocampus"],
            4: [6, "Amygdala"],
            5: [6,"Thalamus"],
        } # where 6 = "SmallStructures"
     
    imgs = [seg, gt]
    for idx, img in enumerate(imgs):
        img_as_np = sitk.GetArrayFromImage(img)
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

# ----------------
# Caller functions
# ----------------
def run_metric_tricks_for_image(
    *,
    img_id,
    seg_raw,
    seg_pp,
    gt_reg,
    tricks_out
):
    @dataclass(frozen=True)
    class MetricTrick:
        name: str
        fn: Callable
        run_on_raw: bool

    tricks = [
        MetricTrick("largestCC", keep_largest_cc, True),
        MetricTrick("shrink", shrink_boundary, True),
        MetricTrick("removeDist", remove_far_voxels, True),
        MetricTrick("morph", mask_dilation_and_erosion, False),
        MetricTrick('relabel', relabel, False)
    ]

    os.makedirs(tricks_out, exist_ok=True)

    for trick in tricks:

        # =========================
        # Apply metric trick on POST-PROCESSED segmentation
        # =========================
        gt_reg_relabeled = None # Placeholder for reduced label granularity experiment - GT needs to be relabeled too
        if trick.name == 'relabel':
            manipulated_pp, gt_reg_relabeled =  __run_metric_trick_experiment(
                seg_pp,
                gt_reg,
                tricks_out,
                f"{img_id}_{trick.name}",
                trick.fn,
                needs_gt_relabel=True
            )
        else:
            manipulated_pp = __run_metric_trick_experiment(
                seg_pp,
                gt_reg,
                tricks_out,
                f"{img_id}_{trick.name}",
                trick.fn,
                needs_gt_relabel= trick.name == 'relabel',
                needs_gt_morph= trick.name == 'morph'
            )
        evaluate(
            manipulated_pp,
            gt_reg if trick.name != 'relabel' else gt_reg_relabeled,
            img_id + f"-TRICK-{trick.name}",
            relabeled= trick.name == 'relabel'
        )

        if trick.run_on_raw:
            # =========================
            # Apply metric trick on RAW segmentation
            # =========================
            manipulated_raw = __run_metric_trick_experiment(
                seg_raw,
                gt_reg,
                tricks_out,
                f"{img_id}_RAW_{trick.name}",
                trick.fn
            )
            evaluate(
                manipulated_raw,
                gt_reg,
                img_id + f"-TRICK-RAW-{trick.name}"
            )


            # =========================
            # Compare RAW vs POST-PROCESSED with trick
            # =========================
            __run_metric_trick_experiment_raw_vs_pp(
                seg_raw,
                seg_pp,
                gt_reg,
                tricks_out,
                f"{img_id}_RAWvsPP_{trick.name}",
                trick.fn
            )


def __run_metric_trick_experiment(seg, gt, outdir, name, trick_fn, needs_gt_relabel=False, needs_gt_morph=False):
    """Run one trick on a segmentation and save results."""
    os.makedirs(outdir, exist_ok=True)

    # Apply trick
    if needs_gt_relabel: # Special case if relabeling is done, GT needs to be relabeled too
        manipulated, gt_relabeled = trick_fn(seg,gt)
    else: 
        manipulated = trick_fn(seg,gt) if needs_gt_morph else trick_fn(seg)
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
                    if x.label == r.label and x.metric == r.metric and x.label != 'SmallStructures'
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


def __run_metric_trick_experiment_raw_vs_pp(
    seg_raw: sitk.Image,
    seg_pp: sitk.Image,
    gt: sitk.Image,
    outdir: str,
    name: str,
    trick_fn
):
    """
    RAW baseline vs (Post-processed + Trick)
    Uses the SAME CSV format as all other metric tricks.
    """
    os.makedirs(outdir, exist_ok=True)

    seg_pp_tricked = trick_fn(seg_pp)
    save_slice(seg_raw, os.path.join(outdir, f"{name}_orig.png"))
    save_slice(seg_pp_tricked, os.path.join(outdir, f"{name}_tricked.png"))

    orig = evaluate(seg_raw, gt)
    trik = evaluate(seg_pp_tricked, gt)

    csv = os.path.join(outdir, f"{name}_metrics.csv")
    with open(csv, "w") as f:
        f.write("Label,Metric,Original,Tricked\n")
        for r in orig:
            match = [x for x in trik if x.metric == r.metric and x.label == r.label and x.label != 'SmallStructures']
            if match:
                f.write(f"{r.label},{r.metric},{r.value},{match[0].value}\n")

    return seg_pp_tricked


# ---------------------------
# Result loading and plotting
# ---------------------------

plot_title_mappings = {
    "largestCC": "Largest Connected Component",
    "shrink": "Boundary Shrink",
    "removeDist": "Remove Distant Components",
    "morph": "Morphological Volume Adjustment",
    "relabel": "Reduced Label Granularity"
}

def load_trick_results(tricks_dir: str, relabeled=False) -> pd.DataFrame:
    """Load all trick CSV results into a single DataFrame."""
    rows = []
    identifier = "_metrics.csv" if not relabeled else "_relabel_metrics.csv"
    print("Files identified:")
    for fname in os.listdir(tricks_dir):
        if not fname.endswith(identifier):
            continue

        print(fname)
        trick_name = fname.replace("_metrics.csv", "")
        subject = trick_name.split("_")[0]
        trick = trick_name.split("_", 1)[1]
        
        if not relabeled and trick == 'relabel': 
            continue # Results for reduced granularity experiments are plotted separately
        df = pd.read_csv(os.path.join(tricks_dir, fname))
        df["Subject"] = subject
        df["Trick"] = trick
        rows.append(df)

    return pd.concat(rows, ignore_index=True)


def plot_trick_summary_boxplot(tricks_df: pd.DataFrame, outdir: str):
    """Plot a summary boxplot for each trick."""
    # Exclude results from reduced label granularity experiment
    tricks_df = tricks_df[(tricks_df["Trick"] != 'relabel') & (tricks_df["Label"] != 'SmallStructures')]

    labels = tricks_df["Label"].unique()
    metrics = tricks_df["Metric"].unique()
    tricks = tricks_df["Trick"].unique()


    for trick in tricks:
        df_t = tricks_df[tricks_df["Trick"] == trick]
        fig, axes = plt.subplots(
            len(labels), len(metrics),
            figsize=(3 * len(metrics), 2.6 * len(labels)),
            squeeze=False,
        )

        for i, label in enumerate(labels):
            df_l = df_t[df_t["Label"] == label]
            for j, metric in enumerate(metrics):
                df_m = df_l[df_l["Metric"] == metric]
                ax = axes[i][j]

                if df_m.empty:
                    ax.set_axis_off()
                    continue

                ax.boxplot([df_m["Original"], df_m["Tricked"]],
                           labels=["Orig", "Adjusted"], showfliers=False)

                if i == 0:
                    ax.set_title(metric)
                if j == 0:
                    ax.set_ylabel(label)
                ax.grid(alpha=0.3)

        plt.suptitle(f"Metric Vulnerability Summary:\n{plot_title_mappings[trick.split('_')[-1]]}", fontsize=15)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(os.path.join(outdir, f"{trick}_summary_boxplot.png"), dpi=150)
        plt.close(fig)

def plot_relabeled_summary_boxplots(tricks_df: pd.DataFrame, outdir: str):
    """
    Boxplots for side-by-side comparison between normal vs reduced granulartiy label performance

    Args:
        tricks_df (pd.DataFrame): results to plot
        outdir (str): directory to store plots
    """    
    metrics_of_interest = ['DICE', 'VOLSMTY', 'HDRFDST']
    metrics = tricks_df["Metric"].unique()

    if not len(tricks_df['Value']):
        print("Empty column in csv, unable to plot relabeled images\' boxplots.")
        return
    
    # Create side-by-side boxplot charts assessing label structures, for all metrics 
    fig, axs = plt.subplots(nrows=len(metrics_of_interest), ncols=1, sharey=False, figsize=(6,4*len(metrics_of_interest)))
    for i, metric in enumerate(metrics_of_interest):
        # Original plot on the left
        results_df = tricks_df[tricks_df["Metric"] == metric]
        labels_of_interest = ['Amygdala', 'Hippocampus', 'Thalamus', 'SmallStructures']
        plot_df = [
            results_df.loc[results_df["Label"] == lbl, "Value"].values
            for lbl in labels_of_interest
        ]
        y_min = 0
        y_max = math.ceil(np.concatenate(plot_df).max())

        axs[i].boxplot(
                        plot_df,
                        labels=labels_of_interest,
                        showfliers=True
                    )
        axs[i].set_ylim(y_min, y_max)
        axs[i].set_ylabel(metric, fontsize=10)
        axs[i].grid(alpha=0.3)
        axs[i].set_title(f"Reduced Label Granularity - {metric}")

        # Rotate all x labels
        for ax in axs:
            ax.tick_params(axis="x", labelrotation=0, labelsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    fig.savefig(os.path.join(outdir, f"relabeled_FULL_summary_boxplot.png"), dpi=150)
    plt.close(fig)

def plot_metric_per_trick_by_class(tricks_df: pd.DataFrame, outdir: str):
    """Plot Original vs Tricked for each class and metric."""
    # Exclude 'SmallStructures' label from Reduced Label Granularity experiment
    tricks_df = tricks_df[(tricks_df["Label"] != "SmallStructures") & (tricks_df["Trick"] != 'relabel')]
    
    classes = sorted(tricks_df["Label"].unique())
    tricks = tricks_df["Trick"].unique()
    metrics = tricks_df["Metric"].unique()

    for trick in tricks:
        df_trick = tricks_df[tricks_df["Trick"] == trick]
        for metric in metrics:
            df_m = df_trick[df_trick["Metric"] == metric]
            if df_m.empty:
                continue

            fig, ax = plt.subplots(figsize=(10, 5))
            orig_data, trick_data = [], []

            for cls in classes:
                df_c = df_m[df_m["Label"] == cls]
                orig_data.append(df_c["Original"].values)
                trick_data.append(df_c["Tricked"].values)

            x = np.arange(len(classes))
            width = 0.35

            bp1 = ax.boxplot(orig_data, positions=x - width / 2, widths=0.3,
                             patch_artist=True, showfliers=False)
            bp2 = ax.boxplot(trick_data, positions=x + width / 2, widths=0.3,
                             patch_artist=True, showfliers=False)

            for box in bp1["boxes"]:
                box.set_facecolor("#4C72B0")
            for box in bp2["boxes"]:
                box.set_facecolor("#DD8452")

            ax.set_xticks(x)
            ax.set_xticklabels(classes, rotation=0)
            ax.set_ylabel(metric)
            ax.set_title(f"{plot_title_mappings[trick.split('_')[-1]]} - {metric}")
            ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["Original", "Adjusted"], loc="best")
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()

            fig.savefig(os.path.join(outdir, f"{trick}_{metric}_by_class.png"), dpi=150)
            plt.close(fig)


