import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pandas as pd

from pymia.evaluation.evaluator import SegmentationEvaluator
from pymia.evaluation.metric import (
    DiceCoefficient,
    JaccardCoefficient,
    AverageDistance,
    VolumeSimilarity,
    HausdorffDistance,
)


# Utility functions
def to_np(img: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(img)


def save_slice(img: sitk.Image, out_path: str, cmap="tab20"):
    """Save mid-axial slice of a SimpleITK image."""
    arr = to_np(img)
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


def evaluate(pred: sitk.Image, gt: sitk.Image):
    """Evaluate a segmentation prediction against ground truth."""
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


# Metric tricks

# -------------------------------------------------------------------------
# Trick 1 — Keep Largest Connected Component
# -------------------------------------------------------------------------
def keep_largest_cc(seg: sitk.Image) -> sitk.Image:
    """For each label, keep only the largest connected component."""
    seg_np = to_np(seg)
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


# -------------------------------------------------------------------------
# Trick 2 — Improved Shrink Boundary
# -------------------------------------------------------------------------
def shrink_boundary(seg: sitk.Image, radius_per_label=None, default_radius=1):
    """
    Suave erosión hacia dentro, específica por etiqueta.
    Usamos BinaryErode con un radio pequeño (en voxels).
    """
    if radius_per_label is None:
        radius_per_label = {
            1: 1, 2: 0, 3: 1, 4: 1, 5: 1
        }

    seg_np = to_np(seg)
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
        eroded_np = to_np(eroded)
        out[eroded_np == 1] = lbl

    out_img = sitk.GetImageFromArray(out.astype(np.uint8))
    out_img.CopyInformation(seg)
    return out_img


# -------------------------------------------------------------------------
# Trick 3 — Remove-Far-Voxels
# -------------------------------------------------------------------------
def remove_far_voxels(
    seg: sitk.Image,
    max_dist_per_label=None,
    min_size_per_label=None,
    default_max_dist=20,
    default_min_size=20
):
    """
    Remove connected components far from the main component,
    using label-dependent distance and size thresholds.
    """
    if max_dist_per_label is None:
        max_dist_per_label = {1: 30, 2: 30, 3: 25, 4: 25, 5: 20}
    if min_size_per_label is None:
        min_size_per_label = {1: 300, 2: 300, 3: 15, 4: 15, 5: 40}

    seg_np = to_np(seg).astype(np.uint8)
    out = np.zeros_like(seg_np, dtype=np.uint8)

    for lbl in np.unique(seg_np):
        if lbl == 0:
            continue
        mask = (seg_np == lbl).astype(np.uint8)
        mask_img = sitk.GetImageFromArray(mask)
        mask_img.CopyInformation(seg)

        cc = sitk.ConnectedComponent(mask_img)
        rel = sitk.RelabelComponent(cc, sortByObjectSize=True)
        rel_np = to_np(rel)

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


# Metric trick experiment functions

def run_metric_trick_experiment(seg, gt, outdir, name, trick_fn):
    """Run one trick on a segmentation and save results."""
    os.makedirs(outdir, exist_ok=True)

    manipulated = trick_fn(seg)
    save_slice(seg, os.path.join(outdir, f"{name}_orig.png"))
    save_slice(manipulated, os.path.join(outdir, f"{name}_tricked.png"))

    orig = evaluate(seg, gt)
    trik = evaluate(manipulated, gt)

    csv = os.path.join(outdir, f"{name}_metrics.csv")
    with open(csv, "w") as f:
        f.write("Label,Metric,Original,Tricked\n")
        for r in orig:
            match = [x for x in trik if x.metric == r.metric and x.label == r.label]
            if match:
                f.write(f"{r.label},{r.metric},{r.value},{match[0].value}\n")

    return manipulated


def run_metric_trick_experiment_raw_vs_pp(
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
            match = [x for x in trik if x.metric == r.metric and x.label == r.label]
            if match:
                f.write(f"{r.label},{r.metric},{r.value},{match[0].value}\n")

    return seg_pp_tricked


# Trick summary CSV and plots

def load_trick_results(tricks_dir: str) -> pd.DataFrame:
    """Load all trick CSV results into a single DataFrame."""
    rows = []
    for f in os.listdir(tricks_dir):
        if not f.endswith("_metrics.csv"):
            continue
        trick_name = f.replace("_metrics.csv", "")
        subject = trick_name.split("_")[0]
        trick = trick_name.split("_", 1)[1]

        df = pd.read_csv(os.path.join(tricks_dir, f))
        df["Subject"] = subject
        df["Trick"] = trick
        rows.append(df)

    return pd.concat(rows, ignore_index=True)


def plot_trick_summary_boxplot(tricks_df: pd.DataFrame, outdir: str):
    """Plot a summary boxplot for each trick."""
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
                           labels=["Orig", "Trick"], showfliers=False)

                if i == 0:
                    ax.set_title(metric)
                if j == 0:
                    ax.set_ylabel(label)
                ax.grid(alpha=0.3)

        plt.suptitle(f"Metric Vulnerability Summary — Trick: {trick}", fontsize=15)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(os.path.join(outdir, f"{trick}_summary_boxplot.png"), dpi=150)
        plt.close(fig)


def plot_metric_per_trick_by_class(tricks_df: pd.DataFrame, outdir: str):
    """Plot Original vs Tricked for each class and metric."""
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
            ax.set_xticklabels(classes, rotation=20)
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} — Trick: {trick}")
            ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["Original", "Tricked"], loc="best")
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()

            fig.savefig(os.path.join(outdir, f"{trick}_{metric}_by_class.png"), dpi=150)
            plt.close(fig)
