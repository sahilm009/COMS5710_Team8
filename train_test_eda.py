import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# ===========================
# CONFIG
# ===========================
BASE_DIR = "IDCIA"  # Contains train, val, test subfolders
OUTPUT_DIR = "eda_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SPLITS = ["train", "val", "test"]


# ===========================
# HELPERS
# ===========================
def load_counts(split):
    """Count number of cells per image in the given split."""
    gt_dir = os.path.join(BASE_DIR, split, "ground_truth")
    rows = []
    for f in os.listdir(gt_dir):
        if f.endswith(".csv"):
            path = os.path.join(gt_dir, f)
            df = pd.read_csv(path)
            rows.append({
                "split": split.capitalize(),
                "filename": f.replace(".csv", ""),
                "num_cells": len(df)
            })
    return pd.DataFrame(rows)


def plot_overlay(split, n=3):
    """Plot a few random overlay samples with red dots marking cells."""
    import random
    img_dir = os.path.join(BASE_DIR, split, "images")
    gt_dir = os.path.join(BASE_DIR, split, "ground_truth")
    img_files = [f for f in os.listdir(img_dir) if f.endswith(".tiff")]
    chosen = random.sample(img_files, min(n, len(img_files)))

    fig, axes = plt.subplots(1, len(chosen), figsize=(5 * len(chosen), 5))
    if len(chosen) == 1:
        axes = [axes]

    for ax, fname in zip(axes, chosen):
        img_path = os.path.join(img_dir, fname)
        gt_path = os.path.join(gt_dir, fname.replace(".tiff", ".csv"))
        img = Image.open(img_path)
        df = pd.read_csv(gt_path)

        ax.imshow(img, cmap="gray")
        if "X" in df.columns and "Y" in df.columns:
            ax.scatter(df["X"], df["Y"], c="red", s=10, label="Cells")
        ax.set_title(f"{split.capitalize()} — {fname}", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"overlay_{split}.png"))
    plt.close()


def plot_heatmap(split):
    """Plot aggregated coordinate heatmap for the given split."""
    gt_dir = os.path.join(BASE_DIR, split, "ground_truth")
    all_x, all_y = [], []

    for f in os.listdir(gt_dir):
        if f.endswith(".csv"):
            df = pd.read_csv(os.path.join(gt_dir, f))
            if "X" in df.columns and "Y" in df.columns:
                all_x.extend(df["X"].dropna())
                all_y.extend(df["Y"].dropna())

    if not all_x:
        print(f"⚠️ No annotations found in {split}. Skipping heatmap.")
        return

    plt.figure(figsize=(6, 6))
    sns.kdeplot(x=all_x, y=all_y, fill=True, cmap="inferno", thresh=0.05)
    plt.title(f"Spatial Density Heatmap — {split.capitalize()}", fontsize=13)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"heatmap_{split}.png"))
    plt.close()


# ===========================
# MAIN EDA
# ===========================
def main():
    print("🔍 Starting detailed EDA on IDCIA dataset...")

    # ---- Load and merge counts ----
    df_all = pd.concat([load_counts(s) for s in SPLITS], ignore_index=True)

    # ---- Summary statistics ----
    summary = (
        df_all.groupby("split")["num_cells"]
        .describe(percentiles=[0.25, 0.5, 0.75])
        .rename(columns={
            "count": "# Images",
            "mean": "Mean # Cells",
            "std": "Std Dev",
            "min": "Min",
            "25%": "25%",
            "50%": "Median",
            "75%": "75%",
            "max": "Max"
        })
        .reset_index()
    )

    # Reorder and format
    summary = summary[
        ["split", "# Images", "Mean # Cells", "Std Dev", "Min", "25%", "Median", "75%", "Max"]
    ]

    # Save to CSV
    summary.to_csv(os.path.join(OUTPUT_DIR, "summary_stats.csv"), index=False)

    # Print in nice console format
    print("\n📊 Summary Statistics:")
    print(summary.to_string(index=False))

    # ---- Boxplot ----
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="split", y="num_cells", data=df_all, palette="coolwarm")
    plt.title("Distribution of Cell Counts per Image (Train / Val / Test)")
    plt.xlabel("Dataset Split")
    plt.ylabel("# of Cells per Image")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "boxplot_cells_per_image.png"))
    plt.close()

    # ---- Histograms ----
    for split in SPLITS:
        plt.figure(figsize=(8, 6))
        sns.histplot(
            df_all[df_all["split"] == split.capitalize()]["num_cells"],
            bins=30,
            kde=True,
            color="skyblue"
        )
        plt.title(f"Histogram of #Cells per Image ({split.capitalize()} Set)")
        plt.xlabel("# of Cells per Image")
        plt.ylabel("Frequency")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"hist_cells_per_image_{split}.png"))
        plt.close()

    # ---- Overlay distributions ----
    plt.figure(figsize=(8, 6))
    for split in SPLITS:
        subset = df_all[df_all["split"] == split.capitalize()]
        sns.kdeplot(subset["num_cells"], label=split.capitalize(), fill=True, alpha=0.3)
    plt.title("Distribution of #Cells per Image Across Splits")
    plt.xlabel("# of Cells per Image")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "overlay_splits.png"))
    plt.close()

    # ---- Visual overlays ----
    for split in SPLITS:
        plot_overlay(split)
        plot_heatmap(split)

    print("\n✅ EDA complete! All visualizations saved in 'eda_outputs/'")


if __name__ == "__main__":
    main()
