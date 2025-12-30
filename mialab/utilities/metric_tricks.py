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

def shrink_boundary(seg: sitk.Image) -> sitk.Image:
    """
    Aggressive per-label inward erosion for metric manipulation.
    Spacing-normalized so erosion ACTUALLY happens.
    """

    radius_per_label = {
        1: 3,  # White Matter
        2: 2,  # Grey Matter
        3: 2,  # Hippocampus
        4: 2,  # Amygdala
        5: 2,  # Thalamus
    }

    seg_np = sitk.GetArrayFromImage(seg)
    out = seg_np.copy()

    for lbl, r in radius_per_label.items():
        mask = (seg_np == lbl).astype(np.uint8)
        if mask.sum() == 0:
            continue

        mask_img = sitk.GetImageFromArray(mask)
        mask_img.CopyInformation(seg)

        mask_img.SetSpacing((1.0, 1.0, 1.0))

        eroded = sitk.BinaryErode(
            mask_img,
            [r, r, r],
            sitk.sitkBall,
            foregroundValue=1
        )

        eroded_np = sitk.GetArrayFromImage(eroded)

        removed = np.sum((mask == 1) & (eroded_np == 0))
        print(f"Label {lbl}: removed {removed} voxels")

        out[(mask == 1) & (eroded_np == 0)] = 0

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

def mask_dilation_and_erosion(seg: sitk.Image, gt: sitk.Image,result_dir=None, img_id=None) -> sitk.Image:
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
    labels = range(1, 6)

    for label in labels:
        pred_mask = (seg_np == label).astype(np.uint8)
        gt_mask = (gt_np == label).astype(np.uint8)

        pred_vol = int(np.sum(pred_mask))
        gt_vol = int(np.sum(gt_mask))
        orig_diff = gt_vol - pred_vol
        orig_abs_diff = abs(orig_diff)

        # default: keep original
        final_mask = pred_mask.copy()

        if orig_abs_diff > 0:
            print(f"Label {label}\nDifference before: ", orig_diff)
            mask_img = sitk.GetImageFromArray(pred_mask)
            mask_img.CopyInformation(seg)

            # decide operation
            morph_op = (
                sitk.BinaryMorphologicalClosing
                if orig_diff > 0
                else sitk.BinaryMorphologicalOpening
            )

            for kernel in ([2, 2, 1], [2, 1, 1]): # try different kernel sizes while avoiding too large of a vol change
                tmp_img = morph_op(mask_img, kernel)
                tmp_mask = sitk.GetArrayFromImage(tmp_img)

                tmp_vol = int(np.sum(tmp_mask))
                tmp_abs_diff = abs(gt_vol - tmp_vol)

                if tmp_abs_diff < orig_abs_diff:
                    final_mask = tmp_mask
                    break  # keep first improvement
            print("Difference after: ", tmp_abs_diff)

        out[final_mask == 1] = label

    out_img = sitk.GetImageFromArray(out.astype(np.uint8))
    out_img.CopyInformation(seg)

    if result_dir and img_id:
        sitk.WriteImage(
            out_img,
            os.path.join(result_dir, img_id + "_TRICKED_volumeMorph.mha"),
            True
        )

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
    seg_pp,
    gt_reg,
    tricks_out
):
    @dataclass(frozen=True)
    class MetricTrick:
        name: str
        fn: Callable

    tricks = [
        MetricTrick("largestCC", keep_largest_cc),
        MetricTrick("shrink", shrink_boundary),
        MetricTrick("removeDist", remove_far_voxels),
        MetricTrick("morph", mask_dilation_and_erosion),
        MetricTrick('relabel', relabel)
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

metric_name_mappings = {
    "HDRFDST": "Hausdorff Distance (95th percentile)",
    "ACURCY": "Accuracy",
    "VOLSMTY": "Volume Similarity",
    "JACRD": "Jaccard Index",
    "AVGDIST": "Average Surface Distance",
    "PRCISON": "Precision",
    "SPCFTY": "Specificity"
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
                           labels=["Original", "After perturbations"], showfliers=False)

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
    tricks_df = tricks_df[tricks_df['Metric'].isin(metrics_of_interest)]
    summary_df = (
        tricks_df
        .groupby(["Metric", "Label"])["Value"]
        .agg(mean="mean", sd="std")
        .reset_index()
    )
    os.makedirs(outdir, exist_ok=True)
    summary_df.to_csv(
        os.path.join(outdir, "relabeled_metric_summary_mean_sd.csv"),
        index=False
    )
    if not len(tricks_df['Value']):
        print("Empty column in csv, unable to plot relabeled images\' boxplots.")
        return
    
    # Create side-by-side boxplot charts assessing label structures, for all metrics 
    
    for metric in metrics_of_interest:
        fig, axs = plt.subplots(figsize=(6,4))
        # Original plot on the left
        results_df = tricks_df[tricks_df["Metric"] == metric]
        labels_of_interest = ['Amygdala', 'Hippocampus', 'Thalamus', 'SmallStructures']
        plot_df = [
            results_df.loc[results_df["Label"] == lbl, "Value"].values
            for lbl in labels_of_interest
        ]
        y_min = 0
        y_max = math.ceil(np.concatenate(plot_df).max())

        axs.boxplot(
                        plot_df,
                        labels=labels_of_interest,
                        showfliers=True
                    )
        axs.set_ylim(y_min, y_max)
        axs.set_ylabel(metric_name_mappings[metric] if metric in metric_name_mappings.keys() else metric, fontsize=10)
        axs.grid(alpha=0.3)
        axs.set_title(f"Reduced Label Granularity - {metric_name_mappings[metric] if metric in metric_name_mappings.keys() else metric}")

        # Rotate all x labels
        axs.tick_params(axis="x", labelrotation=0, labelsize=8)

        fig.tight_layout()

        fig.savefig(os.path.join(outdir, f"relabeled_{metric}_boxplot.png"), dpi=150)
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
            ax.set_ylabel(metric_name_mappings[metric] if metric in metric_name_mappings.keys() else metric)
            ax.set_title(f"{plot_title_mappings[trick.split('_')[-1]]} - {metric_name_mappings[metric] if metric in metric_name_mappings.keys() else metric}")
            ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["Original", "After perturbations"], loc="best")
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()

            fig.savefig(os.path.join(outdir, f"{trick}_{metric}_by_class.png"), dpi=150)
            plt.close(fig)


