"""Evaluate baseline Cellpose segmentation across annotated A172 LIVECell images.

The script runs or reuses Cellpose masks, matches predictions to COCO
annotations with the Hungarian algorithm, and writes cell-level and
image-level segmentation and morphology summaries.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from cellpose import models
from pycocotools.coco import COCO
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from skimage.measure import regionprops
from tqdm import tqdm

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path().resolve().parent
IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "images" / "livecell_train_val_images"
ANNOTATION_FILE = PROJECT_ROOT / "data" / "annotations" / "a172_train.json"
MASK_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# Set to None to process all training images.
MAX_IMAGES = 20
IOU_THRESHOLD = 0.50

# ---------------------------------------------------------------------
# Load annotations
# ---------------------------------------------------------------------

coco = COCO(str(ANNOTATION_FILE))

# Get image IDs in deterministic order
image_ids = sorted(coco.imgs.keys())

if MAX_IMAGES is not None:
    image_ids = image_ids[:MAX_IMAGES]

print(f"Processing {len(image_ids)} A172 training images.")

# ---------------------------------------------------------------------
# Load Cellpose
# ---------------------------------------------------------------------

model = models.CellposeModel(gpu=True, pretrained_model="cpsam_v2")
print("Cellpose model loaded.")

# ---------------------------------------------------------------------
# Calculate IoU matrix
# ---------------------------------------------------------------------

def calculate_iou_matrix(annotations, prediction, image_shape):
    n_gt = len(annotations)
    prediction_labels = np.unique(prediction)
    prediction_labels = prediction_labels[prediction_labels != 0]
    n_pred = len(prediction_labels)

    # Map actual prediction labels to matrix column
    label_to_column = {int(label): i for i, label in enumerate(prediction_labels)}
    pred_areas = {int(label): int(np.sum(prediction == label)) for label in prediction_labels}
    iou_matrix = np.zeros((n_gt, n_pred), dtype=float)

    for gt_index, annotation in enumerate(annotations):
        gt_mask = coco.annToMask(annotation).astype(bool)
        gt_area = int(gt_mask.sum())

        if gt_area == 0:
            continue

        # Look only at prediction labels inside this ground-truth cell.
        labels, intersections = np.unique(prediction[gt_mask], return_counts=True)

        for label, intersection in zip(labels, intersections):
            if label == 0:
                continue

            label = int(label)
            intersection = int(intersection)
            pred_area = pred_areas[label]
            union = gt_area + pred_area - intersection

            if union > 0:
                column = label_to_column[label]
                iou_matrix[gt_index, column] = intersection / union

    return (iou_matrix, prediction_labels)

# ---------------------------------------------------------------------
# Store results
# ---------------------------------------------------------------------

all_cells = []
image_summaries = []

# ---------------------------------------------------------------------
# Process images
# ---------------------------------------------------------------------

for image_id in tqdm(image_ids, desc="Analysing A172 images"):
    metadata = coco.imgs[image_id]
    filename = metadata["file_name"]
    image_path = IMAGE_DIR / filename
    if not image_path.exists():
        print(f"\nWARNING: image missing: {filename}")
        continue

    # ---------------------------------------------------------------
    # Load image
    # ---------------------------------------------------------------

    image = tifffile.imread(image_path)

    # ---------------------------------------------------------------
    # Ground truth
    # ---------------------------------------------------------------

    annotation_ids = coco.getAnnIds(imgIds=[image_id])
    annotations = coco.loadAnns(annotation_ids)
    n_gt = len(annotations)

    # ---------------------------------------------------------------
    # Load or create Cellpose prediction
    # ---------------------------------------------------------------

    mask_path = MASK_DIR / f"{Path(filename).stem}_masks.tif"

    if mask_path.exists():
        prediction = tifffile.imread(mask_path)

    else:
        masks, flows, styles = model.eval(image, channel_axis=None, normalize=True)
        prediction = masks.astype(np.uint16)
        tifffile.imwrite(mask_path, prediction)

    # ---------------------------------------------------------------
    # Calculate IoU matrix
    # ---------------------------------------------------------------

    iou_matrix, prediction_labels = calculate_iou_matrix(annotations, prediction, image.shape)
    n_pred = len(prediction_labels)

    # ---------------------------------------------------------------
    # One-to-one Hungarian matching
    # ---------------------------------------------------------------

    assigned_iou = np.zeros(n_gt, dtype=float)
    assigned_prediction = np.zeros(n_gt, dtype=int)

    if n_gt > 0 and n_pred > 0:
        row_indices, col_indices = linear_sum_assignment(-iou_matrix)

        for row, column in zip(row_indices, col_indices):
            assigned_iou[row] = iou_matrix[row, column]
            assigned_prediction[row] = prediction_labels[column]

    # ---------------------------------------------------------------
    # Determine matches
    # ---------------------------------------------------------------

    matched = assigned_iou >= IOU_THRESHOLD
    true_positives = int(matched.sum())
    false_negatives = n_gt - true_positives
    matched_prediction_labels = set(assigned_prediction[matched])
    matched_prediction_labels.discard(0)
    false_positives = n_pred - len(matched_prediction_labels)

    # ---------------------------------------------------------------
    # Image-level metrics
    # ---------------------------------------------------------------

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    successful_ious = assigned_iou[matched]
    mean_iou = successful_ious.mean() if len(successful_ious) > 0 else np.nan

    # ---------------------------------------------------------------
    # Cell morphology
    # ---------------------------------------------------------------

    cell_records = []

    for cell_index, annotation in enumerate(annotations):
        # Keep each annotation independent.
        mask = coco.annToMask(annotation).astype(np.uint8)
        props = regionprops(mask, intensity_image=image)

        if len(props) != 1:
            print(f"\nWARNING: could not measure " f"cell {cell_index + 1} " f"in {filename}")
            continue

        prop = props[0]

        cell_records.append(
            {
                "image": filename,
                "image_id": image_id,
                "cell_id": cell_index + 1,
                "area": prop.area,
                "perimeter": prop.perimeter,
                "eccentricity": prop.eccentricity,
                "solidity": prop.solidity,
                "extent": prop.extent,
                "centroid_row": prop.centroid[0],
                "centroid_col": prop.centroid[1],
                "assigned_prediction": (assigned_prediction[cell_index]),
                "assigned_iou": (assigned_iou[cell_index]),
                "segmentation_success": (matched[cell_index]),
            }
        )

    all_cells.extend(cell_records)

    # ---------------------------------------------------------------
    # Store image summary
    # ---------------------------------------------------------------

    image_summaries.append(
        {
            "image": filename,
            "image_id": image_id,
            "ground_truth_cells": n_gt,
            "predicted_cells": n_pred,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_iou_successful": mean_iou,
        }
    )

# ---------------------------------------------------------------------
# Convert to DataFrames
# ---------------------------------------------------------------------

cells_df = pd.DataFrame(all_cells)
image_summary_df = pd.DataFrame(image_summaries)

# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

cell_output = OUTPUT_DIR / "a172_multimage_cell_results.csv"
image_output = OUTPUT_DIR / "a172_multimage_image_summary.csv"

cells_df.to_csv(cell_output, index=False)
image_summary_df.to_csv(image_output, index=False)

# ---------------------------------------------------------------------
# Overall metrics
# ---------------------------------------------------------------------

total_tp = int(image_summary_df["true_positives"].sum())
total_fp = int(image_summary_df["false_positives"].sum())
total_fn = int(image_summary_df["false_negatives"].sum())
overall_precision = total_tp / (total_tp + total_fp)
overall_recall = total_tp / (total_tp + total_fn)
overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall)

# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

print("\n")
print("=" * 70)
print("A172 MULTI-IMAGE CELLPOSE ANALYSIS")
print("=" * 70)

print(f"Images analysed: " f"{len(image_summary_df)}")
print(f"Cells analysed: " f"{len(cells_df)}")

print("\nOverall metrics:")
print(f"True positives: {total_tp}")
print(f"False positives: {total_fp}")
print(f"False negatives: {total_fn}")
print(f"Precision: {overall_precision:.3f}")
print(f"Recall: {overall_recall:.3f}")
print(f"F1: {overall_f1:.3f}")
print(f"\nMean assigned IoU: " f"{cells_df['assigned_iou'].mean():.3f}")
print(f"Median assigned IoU: " f"{cells_df['assigned_iou'].median():.3f}")

print("\nImage-level summary:")
print(image_summary_df[["precision", "recall", "f1", "mean_iou_successful"]].describe())

print(f"\nSaved cell results to:\n" f"{cell_output}")
print(f"\nSaved image results to:\n" f"{image_output}")

df = image_summary_df

# ---------------------------------------------------------------------
# Add useful derived variables
# ---------------------------------------------------------------------

# All LIVECell images are 520 x 704 in our analysis.
IMAGE_AREA = 520 * 704

df["cell_density_per_100k_px"] = df["ground_truth_cells"] / IMAGE_AREA * 100_000

df["prediction_to_truth_ratio"] = df["predicted_cells"] / df["ground_truth_cells"]

df["false_negative_rate"] = df["false_negatives"] / df["ground_truth_cells"]

# ---------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------

metrics = ["precision", "recall", "f1", "mean_iou_successful"]

print("\nSpearman correlations with ground-truth cell count:")
print("-" * 55)

for metric in metrics:
    rho, p_value = spearmanr(df["ground_truth_cells"], df[metric])

    print(f"{metric:25s}" f" rho = {rho: .3f}" f"   p = {p_value:.4g}")

print("\nSpearman correlation with false-negative rate:")

rho, p_value = spearmanr(df["ground_truth_cells"], df["false_negative_rate"])

print(f"rho = {rho:.3f}" f"   p = {p_value:.4g}")

# ---------------------------------------------------------------------
# Show easiest / hardest images
# ---------------------------------------------------------------------

print("\nHardest images by F1:")
print("-" * 80)

print(
    df[["image", "ground_truth_cells", "predicted_cells", "precision", "recall", "f1", "prediction_to_truth_ratio"]]
    .sort_values("f1")
    .head(10)
    .to_string(index=False)
)

# ---------------------------------------------------------------------
# Save derived table
# ---------------------------------------------------------------------

output_file = OUTPUT_DIR / "a172_density_analysis.csv"

df.to_csv(output_file, index=False)

print(f"\nSaved analysis to:\n{output_file}")

# ---------------------------------------------------------------------
# Plot 1: Cell count vs recall
# ---------------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(df["ground_truth_cells"], df["recall"], alpha=0.75)

plt.xlabel("Number of ground-truth cells")

plt.ylabel("Recall")

plt.title("Cell density vs Cellpose recall")

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "cell_density_vs_recall.png", dpi=200, bbox_inches="tight")

plt.show()

# ---------------------------------------------------------------------
# Plot 2: Cell count vs F1
# ---------------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(df["ground_truth_cells"], df["f1"], alpha=0.75)

plt.xlabel("Number of ground-truth cells")

plt.ylabel("F1 score")

plt.title("Cell density vs Cellpose F1")

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "cell_density_vs_f1.png", dpi=200, bbox_inches="tight")

plt.show()

# ---------------------------------------------------------------------
# Plot 3: Prediction / truth ratio vs recall
# ---------------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(df["prediction_to_truth_ratio"], df["recall"], alpha=0.75)

plt.axvline(1.0, linestyle="--")

plt.xlabel("Predicted cells / ground-truth cells")

plt.ylabel("Recall")

plt.title("Prediction-to-truth ratio vs recall")

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "prediction_truth_ratio_vs_recall.png", dpi=200, bbox_inches="tight")

plt.show()

NEIGHBOUR_RADIUS = 50
cells = cells_df

# ---------------------------------------------------------------------
# Calculate local crowding
# ---------------------------------------------------------------------

local_neighbour_counts = []
nearest_neighbour_distances = []

for image_name, group in cells.groupby("image"):
    coordinates = group[["centroid_row", "centroid_col"]].to_numpy()
    tree = cKDTree(coordinates)

    # ---------------------------------------------------------------
    # Number of neighbours within the chosen radius
    #
    # query_ball_point includes the cell itself, so subtract 1.
    # ---------------------------------------------------------------

    neighbour_lists = tree.query_ball_point(coordinates, r=NEIGHBOUR_RADIUS)
    counts = np.array([len(neighbours) - 1 for neighbours in neighbour_lists])

    local_neighbour_counts.extend(counts)

    # ---------------------------------------------------------------
    # Nearest-neighbour distance
    #
    # k=2 because the nearest point to a cell is itself.
    # ---------------------------------------------------------------

    if len(group) > 1:
        distances, _ = tree.query(coordinates, k=2)

        nearest_distances = distances[:, 1]

    else:
        nearest_distances = np.array([np.nan])

    nearest_neighbour_distances.extend(nearest_distances)

cells["neighbours_within_50px"] = local_neighbour_counts

cells["nearest_neighbour_distance"] = nearest_neighbour_distances

# ---------------------------------------------------------------------
# Basic summary
# ---------------------------------------------------------------------

print("\nLocal crowding summary:")

print(cells[["neighbours_within_50px", "nearest_neighbour_distance"]].describe())

# ---------------------------------------------------------------------
# Correlation with IoU
# ---------------------------------------------------------------------

print("\nSpearman correlations:")

rho, p = spearmanr(cells["neighbours_within_50px"], cells["assigned_iou"])

print(f"Neighbour count vs IoU:" f" rho = {rho:.3f}" f"   p = {p:.4g}")

rho, p = spearmanr(cells["nearest_neighbour_distance"], cells["assigned_iou"])

print(f"Nearest-neighbour distance vs IoU:" f" rho = {rho:.3f}" f"   p = {p:.4g}")

rho, p = spearmanr(cells["neighbours_within_50px"], cells["segmentation_success"])

print(f"Neighbour count vs segmentation success:" f" rho = {rho:.3f}" f"   p = {p:.4g}")

# ---------------------------------------------------------------------
# Compare successful and failed cells
# ---------------------------------------------------------------------

successful = cells[cells["segmentation_success"]]
failed = cells[~cells["segmentation_success"]]

print("\nNeighbour count by segmentation outcome:")

print(f"Successful cells:" f" median = " f"{successful['neighbours_within_50px'].median():.1f}")

print(f"Failed cells:" f" median = " f"{failed['neighbours_within_50px'].median():.1f}")

print("\nNearest-neighbour distance:")

print(f"Successful cells:" f" median = " f"{successful['nearest_neighbour_distance'].median():.2f}")

print(f"Failed cells:" f" median = " f"{failed['nearest_neighbour_distance'].median():.2f}")

# ---------------------------------------------------------------------
# Add crowding bins
# ---------------------------------------------------------------------

# Quartiles give us four groups with approximately equal numbers of cells.

cells["crowding_quartile"] = pd.qcut(
    cells["neighbours_within_50px"],
    q=4,
    labels=["Q1: least crowded", "Q2", "Q3", "Q4: most crowded"],
    duplicates="drop",
)

crowding_summary = cells.groupby("crowding_quartile", observed=True)[
    ["assigned_iou", "segmentation_success", "neighbours_within_50px"]
].agg({"assigned_iou": ["mean", "median"], "segmentation_success": "mean", "neighbours_within_50px": "mean"})

print("\nPerformance by crowding quartile:")

print(crowding_summary)

# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

output_file = OUTPUT_DIR / "a172_local_crowding_results.csv"

cells.to_csv(output_file, index=False)

print(f"\nSaved results to:\n" f"{output_file}")

# ---------------------------------------------------------------------
# Plot 1: local neighbours vs IoU
# ---------------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(cells["neighbours_within_50px"], cells["assigned_iou"], alpha=0.35)

plt.xlabel("Number of neighbouring cells within 50 px")

plt.ylabel("Cellpose IoU")

plt.title("Local cell crowding vs segmentation quality")

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "local_crowding_vs_iou.png", dpi=200, bbox_inches="tight")

plt.show()

# ---------------------------------------------------------------------
# Plot 2: nearest-neighbour distance vs IoU
# ---------------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(cells["nearest_neighbour_distance"], cells["assigned_iou"], alpha=0.35)

plt.xlabel("Nearest-neighbour distance (pixels)")

plt.ylabel("Cellpose IoU")

plt.title("Cell spacing vs segmentation quality")

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "nearest_neighbour_vs_iou.png", dpi=200, bbox_inches="tight")

plt.show()

# ---------------------------------------------------------------------
# Plot 3: crowding quartile vs IoU
# ---------------------------------------------------------------------

plot_data = cells.groupby("crowding_quartile", observed=True)["assigned_iou"].median()

plt.figure(figsize=(8, 6))

plt.plot(range(len(plot_data)), plot_data.values, marker="o")

plt.xticks(range(len(plot_data)), plot_data.index, rotation=20, ha="right")

plt.ylabel("Median Cellpose IoU")

plt.xlabel("Local crowding")

plt.title("Segmentation quality across local crowding levels")

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "crowding_quartiles_vs_iou.png", dpi=200, bbox_inches="tight")

plt.show()

images = df

# ---------------------------------------------------------------------
# Create image-level morphology summary
# ---------------------------------------------------------------------

morphology_summary = (
    cells.groupby("image")
    .agg(
        median_area=("area", "median"),
        mean_area=("area", "mean"),
        median_solidity=("solidity", "median"),
        mean_solidity=("solidity", "mean"),
        median_eccentricity=("eccentricity", "median"),
        median_extent=("extent", "median"),
        median_neighbours=("neighbours_within_50px", "median"),
        mean_neighbours=("neighbours_within_50px", "mean"),
        median_nearest_distance=("nearest_neighbour_distance", "median"),
        mean_nearest_distance=("nearest_neighbour_distance", "mean"),
    )
    .reset_index()
)

# ---------------------------------------------------------------------
# Merge with Cellpose image-level metrics
# ---------------------------------------------------------------------

df = images.merge(morphology_summary, on="image", how="inner")

# ---------------------------------------------------------------------
# Display table
# ---------------------------------------------------------------------

print("\nCombined image-level data:")

print(
    df[
        [
            "image",
            "ground_truth_cells",
            "median_area",
            "median_solidity",
            "median_eccentricity",
            "median_neighbours",
            "recall",
            "f1",
        ]
    ].to_string(index=False)
)

# ---------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------

predictors = [
    "ground_truth_cells",
    "median_area",
    "mean_area",
    "median_solidity",
    "median_eccentricity",
    "median_extent",
    "median_neighbours",
    "median_nearest_distance",
]

outcomes = ["precision", "recall", "f1", "mean_iou_successful"]

print("\n")
print("=" * 75)
print("IMAGE-LEVEL SPEARMAN CORRELATIONS")
print("=" * 75)

for outcome in outcomes:
    print(f"\nOutcome: {outcome}")
    print("-" * 50)

    for predictor in predictors:
        rho, p_value = spearmanr(df[predictor], df[outcome])

        print(f"{predictor:30s}" f" rho = {rho: .3f}" f"   p = {p_value:.4g}")

# ---------------------------------------------------------------------
# Rank predictors by absolute association with F1
# ---------------------------------------------------------------------

correlations = []

for predictor in predictors:
    rho, p_value = spearmanr(df[predictor], df["f1"])

    correlations.append({"predictor": predictor, "rho": rho, "p_value": p_value, "abs_rho": abs(rho)})

correlation_df = pd.DataFrame(correlations).sort_values("abs_rho", ascending=False)

print("\n")
print("=" * 75)
print("PREDICTORS OF IMAGE-LEVEL F1")
print("=" * 75)

print(correlation_df[["predictor", "rho", "p_value"]].to_string(index=False))

# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

output_file = OUTPUT_DIR / "a172_image_level_analysis.csv"

df.to_csv(output_file, index=False)

print(f"\nSaved combined analysis to:\n" f"{output_file}")

# ---------------------------------------------------------------------
# Plot: median cell area vs F1
# ---------------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(df["median_area"], df["f1"], alpha=0.8)

plt.xlabel("Median cell area (pixels)")

plt.ylabel("Cellpose F1")

plt.title("Cell size vs image-level segmentation performance")

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "median_cell_area_vs_f1.png", dpi=200, bbox_inches="tight")

plt.show()

# ---------------------------------------------------------------------
# Plot: cell count vs median cell area
# ---------------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(df["ground_truth_cells"], df["median_area"], alpha=0.8)

plt.xlabel("Number of ground-truth cells")

plt.ylabel("Median cell area (pixels)")

plt.title("Cell number vs cell size across microscopy fields")

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "cell_count_vs_median_area.png", dpi=200, bbox_inches="tight")

plt.show()

# ---------------------------------------------------------------------
# Plot: solidity vs F1
# ---------------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(df["median_solidity"], df["f1"], alpha=0.8)

plt.xlabel("Median cell solidity")

plt.ylabel("Cellpose F1")

plt.title("Cell morphology vs image-level segmentation performance")

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "median_solidity_vs_f1.png", dpi=200, bbox_inches="tight")

plt.show()