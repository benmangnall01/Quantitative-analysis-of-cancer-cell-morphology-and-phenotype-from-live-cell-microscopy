# ---------------------------------------------------------------------
# Load packages
# ---------------------------------------------------------------------

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tifffile
from cellpose import models
from pycocotools.coco import COCO
from scipy.optimize import linear_sum_assignment
from matplotlib.patches import Polygon
from skimage.measure import regionprops_table
from scipy.stats import mannwhitneyu

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path().resolve().parent

# Define paths
IMAGE_DIR = (PROJECT_ROOT / "data" / "raw" / "images" / "livecell_train_val_images")
ANNOTATION_FILE = (PROJECT_ROOT / "data" / "annotations" / "a172_train.json")
MASK_FILE = (PROJECT_ROOT / "data" / "processed" / "A172_Phase_D7_1_02d04h00m_3_cellpose_masks.tif")
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------------
# Load raw image 
# ---------------------------------------------------------------------

# Define some parameters
IOU_THRESHOLD = 0.50
FILENAME = "A172_Phase_D7_1_02d04h00m_3.tif"

# Load image
image_path = IMAGE_DIR / FILENAME
image = tifffile.imread(image_path)

print(f"Image shape: {image.shape}")

# ---------------------------------------------------------------------
# Load ground-truth COCO annotations
# ---------------------------------------------------------------------

# Load annotations
coco = COCO(str(ANNOTATION_FILE))

# Get the image
matching_image_ids = [
    image_id
    for image_id, metadata in coco.imgs.items()
    if metadata["file_name"] == FILENAME
]

image_id = matching_image_ids[0]

annotation_ids = coco.getAnnIds(imgIds=[image_id])
annotations = coco.loadAnns(annotation_ids)

print(f"Ground-truth cells: {len(annotations)}")

# ---------------------------------------------------------------------
# Convert COCO annotations into an instance-label image
# ---------------------------------------------------------------------

ground_truth = np.zeros(image.shape, dtype=np.uint16)
for label, annotation in enumerate(annotations, start=1):
    mask = coco.annToMask(annotation)
    ground_truth[mask > 0] = label

n_ground_truth = len(annotations)
print(f"Ground-truth labels: " f"{len(np.unique(ground_truth)) - 1}")

# ---------------------------------------------------------------------
# Calculate pairwise IoU efficiently
# ---------------------------------------------------------------------

# Reload predictions 
prediction = tifffile.imread(MASK_FILE)
n_predictions = int(prediction.max())
print(f"Cellpose predictions: {n_predictions}")
print("\nCalculating pairwise IoU...")

n_gt = n_ground_truth
n_pred = n_predictions

# Number of pixels belonging to each object
gt_areas = np.bincount(ground_truth.ravel(), minlength=n_gt + 1)
pred_areas = np.bincount(prediction.ravel(), minlength=n_pred + 1)

# For every pixel, encode the pair:
#  ground_truth_label * (n_pred + 1) + predicted_label

# Counting these codes gives the number of intersecting pixels between every ground-truth / prediction pair.
pair_codes = (ground_truth.astype(np.int64) * (n_pred + 1) + prediction.astype(np.int64))
pair_counts = np.bincount(pair_codes.ravel(), minlength=(n_gt + 1) * (n_pred + 1),)
intersection_matrix = pair_counts.reshape((n_gt + 1, n_pred + 1))

# Ignore background (label 0)
intersections = intersection_matrix[1:, 1:]

# Calculate union: union = area(A) + area(B) - intersection
unions = (gt_areas[1:, None] + pred_areas[None, 1:] - intersections)

iou_matrix = np.divide(intersections, unions, out=np.zeros_like(intersections, dtype=float), where=unions > 0)

# Hungarian algorithm finds the assignment that maximises total IoU.
row_indices, col_indices = linear_sum_assignment(-iou_matrix)

# ---------------------------------------------------------------------
# Determine true positives, false positives and false negatives
# ---------------------------------------------------------------------

matches = []

for gt_idx, pred_idx in zip(row_indices, col_indices):
    iou = iou_matrix[gt_idx, pred_idx]
    matches.append(
        {   "ground_truth_label": gt_idx + 1,
            "prediction_label": pred_idx + 1,
            "iou": iou,
            "matched": iou >= IOU_THRESHOLD,
        }
    )

matches_df = pd.DataFrame(matches)
true_positives = int(matches_df["matched"].sum())
false_negatives = n_gt - true_positives
false_positives = n_pred - true_positives

# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

precision = (true_positives / (true_positives + false_positives))
recall = (true_positives / (true_positives + false_negatives))
f1 = (2 * precision * recall / (precision + recall))
matched_ious = matches_df.loc[matches_df["matched"], "iou"]
mean_matched_iou = matched_ious.mean()
median_matched_iou = matched_ious.median()

print(f"Ground-truth cells:       {n_gt}")
print(f"Predicted cells:          {n_pred}")

print(f"\nIoU threshold:             {IOU_THRESHOLD:.2f}")

print(f"\nTrue positives:            {true_positives}")
print(f"False positives:           {false_positives}")
print(f"False negatives:           {false_negatives}")

print(f"\nPrecision:                 {precision:.3f}")
print(f"Recall:                    {recall:.3f}")
print(f"F1 score:                  {f1:.3f}")

print(f"\nMean matched IoU:          {mean_matched_iou:.3f}")
print(f"Median matched IoU:       {median_matched_iou:.3f}")

# ---------------------------------------------------------------------
# Save object-level results
# ---------------------------------------------------------------------

results_path = OUTPUT_DIR / "cellpose_object_matches.csv"
matches_df.to_csv(results_path, index=False)

# ---------------------------------------------------------------------
# Diagnostic visualisation
# ---------------------------------------------------------------------

print("\nLowest-IoU matches:")
print(matches_df.sort_values("iou").head(10).to_string(index=False))

# Ground-truth objects that failed to match
matched_gt_labels = set(matches_df.loc[matches_df["matched"], "ground_truth_label"])
matched_pred_labels = set(matches_df.loc[matches_df["matched"], "prediction_label"])

# Ground-truth cells with no acceptable prediction
missed_gt = [
    label
    for label in range(1, n_gt + 1)
    if label not in matched_gt_labels
]

# Predicted cells with no acceptable ground-truth match
false_positive_pred = [
    label
    for label in range(1, n_pred + 1)
    if label not in matched_pred_labels
]

fig, axes = plt.subplots(1, 3, figsize=(18, 7))

# Raw image
axes[0].imshow(image, cmap="gray")
axes[0].set_title("Raw A172 phase-contrast image")
axes[0].axis("off")

# Ground truth
axes[1].imshow(image, cmap="gray")
for label in range(1, n_gt + 1):
    mask = ground_truth == label
    axes[1].contour(mask, levels=[0.5], linewidths=0.5)

axes[1].set_title(f"Ground truth\n{n_gt} cells")
axes[1].axis("off")

# Cellpose
axes[2].imshow(image, cmap="gray")
for label in range(1, n_pred + 1):
    mask = prediction == label
    axes[2].contour(mask, levels=[0.5], linewidths=0.5)

axes[2].set_title(f"Cellpose\n{n_pred} cells")
axes[2].axis("off")
plt.tight_layout()

# Save
figure_path = OUTPUT_DIR / "cellpose_evaluation_overview.png"
plt.savefig(figure_path, dpi=200, bbox_inches="tight",)
plt.show()

print(f"\nSaved overview figure to:")
print(figure_path)

# Final summary
print("\nDifficult cases:")
print(f"  Missed ground-truth cells: {len(missed_gt)}")
print(f"  Unmatched predictions:     {len(false_positive_pred)}")

# ---------------------------------------------------------------------
# Diagnose failures
# ---------------------------------------------------------------------

# Get best predictions for every ground truth cell
best_pred_for_gt = np.argmax(iou_matrix, axis=1)
best_iou_for_gt = np.max(iou_matrix, axis=1)

# Best ground-truth cell for every prediction
best_gt_for_pred = np.argmax(iou_matrix, axis=0)
best_iou_for_pred = np.max(iou_matrix, axis=0)

# Identify missed ground-truth cells
missed_gt = np.where(best_iou_for_gt < IOU_THRESHOLD)[0] + 1

# Identify unmatched predictions
unmatched_predictions = np.where(best_iou_for_pred < IOU_THRESHOLD)[0] + 1
print(f"\nGround-truth cells without a match: " f"{len(missed_gt)}")
print(f"Predictions without a match: " f"{len(unmatched_predictions)}")

# Print the worst actual ground-truth cases
gt_failure_table = pd.DataFrame(
    {
        "ground_truth_label": np.arange(1, n_gt + 1),
        "best_prediction": best_pred_for_gt + 1,
        "best_iou": best_iou_for_gt,
        "matched": best_iou_for_gt >= IOU_THRESHOLD,
    }
)

print("\nWorst ground-truth cases:")
print(gt_failure_table.sort_values("best_iou").head(15).to_string(index=False))

# Print the worst actual prediction cases
prediction_failure_table = pd.DataFrame(
    {
        "prediction_label": np.arange(1, n_pred + 1),
        "best_ground_truth": best_gt_for_pred + 1,
        "best_iou": best_iou_for_pred,
        "matched": best_iou_for_pred >= IOU_THRESHOLD,
    }
)

print("\nWorst Cellpose predictions:")
print(prediction_failure_table.sort_values("best_iou").head(15).to_string(index=False))

# Save tables
gt_failure_table.to_csv(OUTPUT_DIR / "ground_truth_match_analysis.csv", index=False)
prediction_failure_table.to_csv(OUTPUT_DIR / "prediction_match_analysis.csv", index=False)

# Create failure masks
missed_mask = np.isin(ground_truth, missed_gt)
false_positive_mask = np.isin(prediction, unmatched_predictions)

# Visualise failure cases
fig, axes = plt.subplots(1, 3, figsize=(18, 7))

# Ground-truth cells
axes[0].imshow(image, cmap="gray")
axes[0].contour(ground_truth > 0, levels=[0.5], linewidths=0.5)
axes[0].set_title(f"Ground truth\n{n_gt} cells")
axes[0].axis("off")

# Missed ground-truth cells
axes[1].imshow(image, cmap="gray")
axes[1].contour(missed_mask, levels=[0.5], linewidths=1.5)
axes[1].set_title(f"Ground-truth cells without match\n" f"{len(missed_gt)} cells")
axes[1].axis("off")

# Unmatched predictions
axes[2].imshow(image, cmap="gray")
axes[2].contour(false_positive_mask, levels=[0.5], linewidths=1.5)
axes[2].set_title(f"Cellpose predictions without match\n" f"{len(unmatched_predictions)} cells")
axes[2].axis("off")

plt.tight_layout()

figure_path = (OUTPUT_DIR / "cellpose_failure_analysis.png")

plt.savefig(figure_path, dpi=200, bbox_inches="tight")
plt.show()

print(f"\nSaved failure figure to:\n{figure_path}")

# ---------------------------------------------------------------------
# Extract morphology measurements
# ---------------------------------------------------------------------

properties = [
    "label",
    "area",
    "perimeter",
    "eccentricity",
    "solidity",
    "extent",
    "centroid",
]

measurements = regionprops_table(ground_truth, intensity_image=image, properties=properties)
cells = pd.DataFrame(measurements)

# Add segmentation results
cells["best_prediction"] = best_pred_for_gt + 1
cells["best_iou"] = best_iou_for_gt

cells["matched"] = (cells["best_iou"] >= IOU_THRESHOLD)

# A simple binary outcome:
# 1 = Cellpose produced an acceptable segmentation
# 0 = it did not

cells["segmentation_success"] = (cells["matched"].astype(int))

# Display summary
print("\nSegmentation success:")
print(cells["segmentation_success"].value_counts().sort_index())
print("\nMean morphology by segmentation outcome:")

summary = (
    cells.groupby("matched")[
        [
            "area",
            "perimeter",
            "eccentricity",
            "solidity",
            "extent",
            "best_iou",
        ]].mean())

print(summary)

# Statistical comparisons
successful = cells[cells["matched"]]["area"]
failed = cells[~cells["matched"]]["area"]

statistic, p_value = mannwhitneyu(successful, failed, alternative="two-sided")

print("\nArea comparison")
print("----------------")
print(f"Median successful area: {successful.median():.1f}")
print(f"Median failed area:     {failed.median():.1f}")
print(f"Mann-Whitney U p-value:  {p_value:.4g}")


# Correlation between morphology and IoU
print("\nSpearman correlations with best IoU:")
morphology_features = ["area", "perimeter", "eccentricity", "solidity", "extent"]

correlations = []

for feature in morphology_features:
    rho = cells[feature].corr(cells["best_iou"], method="spearman")
    correlations.append({"feature": feature, "spearman_rho": rho})

correlation_df = pd.DataFrame(correlations)
print(correlation_df.to_string(index=False))

# Save complete dataset
csv_path = (OUTPUT_DIR / "cell_morphology_segmentation_results.csv")
cells.to_csv(csv_path, index=False)
print(f"\nSaved results to:\n{csv_path}")

# ---------------------------------------------------------------------
# Some plots
# ---------------------------------------------------------------------

# Area vs IOU
plt.figure(figsize=(8, 6))
plt.scatter(cells["area"], cells["best_iou"], alpha=0.6)
plt.axhline(IOU_THRESHOLD, linestyle="--")
plt.xlabel("Ground-truth cell area (pixels)")
plt.ylabel("Best Cellpose IoU")
plt.title("Cell size vs Cellpose segmentation quality")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "area_vs_cellpose_iou.png", dpi=200, bbox_inches="tight")
plt.show()

# Eccentricity vs IoU
plt.figure(figsize=(8, 6))
plt.scatter(cells["eccentricity"], cells["best_iou"], alpha=0.6)
plt.axhline(IOU_THRESHOLD, linestyle="--")
plt.xlabel("Ground-truth eccentricity")
plt.ylabel("Best Cellpose IoU")
plt.title("Cell elongation vs Cellpose segmentation quality")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "eccentricity_vs_cellpose_iou.png", dpi=200, bbox_inches="tight")
plt.show()

# Solidity vs IoU
plt.figure(figsize=(8, 6))
plt.scatter(cells["solidity"], cells["best_iou"], alpha=0.6)
plt.axhline(IOU_THRESHOLD, linestyle="--")
plt.xlabel("Ground-truth solidity")
plt.ylabel("Best Cellpose IoU")
plt.title("Cell irregularity vs Cellpose segmentation quality")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "solidity_vs_cellpose_iou.png", dpi=200, bbox_inches="tight")
plt.show()