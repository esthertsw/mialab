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

def to_np(img: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(img)


# -------------------------------------------------------------------------
# Trick 1 — Keep Largest Connected Component
# -------------------------------------------------------------------------
def keep_largest_cc(seg: sitk.Image) -> sitk.Image:
    " For each label, keep only the largest connected component. "
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
def shrink_boundary(seg: sitk.Image,
                    radius_per_label=None,
                    default_radius=1):
    """
    Suave erosión hacia dentro, específica por etiqueta.
    Usamos BinaryErode con un radio pequeño (en voxels).
    """

    if radius_per_label is None:
        radius_per_label = {
            1: 1,   # White matter  → erodes 1 voxel
            2: 0,   # Grey matter   → dont erode
            3: 1,   # Hippocampus
            4: 1,   # Amygdala
            5: 1,   # Thalamus
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

        r_int = int(round(r))
        radius_vec = [r_int] * seg.GetDimension()

        eroded = sitk.BinaryErode(
            mask_img,
            radius_vec,
            sitk.sitkBall,
            1                  # foregroundValue
        )

        eroded_np = sitk.GetArrayFromImage(eroded)
        out[eroded_np == 1] = lbl

    out_img = sitk.GetImageFromArray(out.astype(np.uint8))
    out_img.CopyInformation(seg)
    return out_img


# -------------------------------------------------------------------------
# Trick 3 — Remove-Far-Voxels
# -------------------------------------------------------------------------
def remove_far_voxels(seg: sitk.Image, frac_per_label=None, default_frac=0.8):
    """
    Controlled compacting toward the centroid. 
    """

    if frac_per_label is None:
        frac_per_label = {
            1: 0.95,  # WM
            2: 0.95,  # GM
            3: 0.75,  # Hipp
            4: 0.75,  # Amy
            5: 0.80,  # Tha
        }

    seg_np = to_np(seg)
    out = np.zeros_like(seg_np, dtype=np.uint8)

    labels = [l for l in np.unique(seg_np) if l != 0]

    for lbl in labels:
        coords = np.column_stack(np.where(seg_np == lbl))
        if len(coords) == 0:
            continue

        center = coords.mean(axis=0)
        d = np.linalg.norm(coords - center, axis=1)

        max_d = d.max()
        thr = frac_per_label.get(lbl, default_frac) * max_d
        keep_mask = d <= thr

        for c in coords[keep_mask]:
            out[tuple(c)] = lbl

    out_img = sitk.GetImageFromArray(out)
    out_img.CopyInformation(seg)
    return out_img


def get_metrics():
    return [
        DiceCoefficient(),
        JaccardCoefficient(),
        VolumeSimilarity(),
        AverageDistance(),
        HausdorffDistance(percentile=95, metric="HDRFDST"),
    ]


def evaluate(pred: sitk.Image, gt: sitk.Image):
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


# Save mid slice
def save_slice(img: sitk.Image, out_path: str, cmap="tab20"):
    arr = to_np(img)
    mid = arr.shape[0] // 2

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure(figsize=(5, 5))
    plt.imshow(arr[mid], cmap=cmap)
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight", dpi=130)
    plt.close()


# Execute one trick experiment
def run_metric_trick_experiment(seg, gt, outdir, name, trick_fn):

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


# Load all trick CSVs
def load_trick_results(tricks_dir: str) -> pd.DataFrame:
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
    labels = tricks_df["Label"].unique()
    metrics = tricks_df["Metric"].unique()
    tricks = tricks_df["Trick"].unique()

    for trick in tricks:
        df_t = tricks_df[tricks_df["Trick"] == trick]

        fig, axes = plt.subplots(
            len(labels),
            len(metrics),
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

                ax.boxplot(
                    [df_m["Original"], df_m["Tricked"]],
                    labels=["Orig", "Trick"],
                    showfliers=False,
                )

                if i == 0:
                    ax.set_title(metric)
                if j == 0:
                    ax.set_ylabel(label)

                ax.grid(alpha=0.3)

        plt.suptitle(f"Metric Vulnerability Summary — Trick: {trick}", fontsize=15)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(os.path.join(outdir, f"{trick}_summary_boxplot.png"), dpi=150)
        plt.close(fig)
