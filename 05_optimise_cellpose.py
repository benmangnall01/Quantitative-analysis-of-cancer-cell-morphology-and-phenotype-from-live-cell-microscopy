"""Systematically optimise Cellpose for A172 LIVECell images.

This script uses a tuning/validation split so that parameters are selected on
one set of annotated images and tested on different images.  It performs a
two-stage coordinate search:

1. Search cell-probability and flow thresholds at the native image scale.
2. Search cell diameter using the best thresholds from stage 1.

The best tuning configuration is then compared with the Cellpose defaults on
the held-out validation images.  Validation masks, result tables, a JSON file
containing the selected parameters, and diagnostic figures are saved.
"""

import json
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from cellpose import models
from pycocotools.coco import COCO
from scipy.optimize import linear_sum_assignment
from skimage.segmentation import find_boundaries
from tqdm import tqdm

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path().resolve().parent
IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "images" / "livecell_train_val_images"
ANNOTATION_FILE = PROJECT_ROOT / "data" / "annotations" / "a172_train.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "cellpose_optimization"
MASK_DIR = PROJECT_ROOT / "data" / "processed" / "cellpose_optimized_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MASK_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# Start with a small experiment.  Set MAX_IMAGES to None after the script works end-to-end and you are ready for a longer parameter search.
MAX_IMAGES = 20
TUNING_FRACTION = 0.60
RANDOM_SEED = 42
IOU_THRESHOLD = 0.50
MODEL_NAME = "cpsam_v2"
USE_GPU = True

# Current Cellpose defaults used as the baseline.
BASELINE_PARAMETERS = {"diameter": None, "cellprob_threshold": 0.0, "flow_threshold": 0.4, "min_size": 15}

# Stage 1 searches mask detection and mask-quality thresholds.
CELLPROB_THRESHOLDS = [-1.0, 0.0, 1.0]
FLOW_THRESHOLDS = [0.2, 0.4, 0.6]

# Stage 2 searches scale using the best thresholds from stage 1.
DIAMETERS = [20, 30, 40, 50]

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def create_ground_truth(coco, annotations, image_shape):
    """Convert COCO annotations into an instance-label image."""

    ground_truth = np.zeros(image_shape, dtype=np.uint32)

    for label, annotation in enumerate(annotations, start=1):
        mask = coco.annToMask(annotation)
        ground_truth[mask > 0] = label

    return ground_truth

def calculate_iou_matrix(ground_truth, prediction):
    """Calculate all ground-truth/prediction IoUs efficiently."""

    n_gt = int(ground_truth.max())
    prediction_labels = np.unique(prediction)
    prediction_labels = prediction_labels[prediction_labels != 0]
    n_pred = len(prediction_labels)

    if n_gt == 0 or n_pred == 0:
        return np.zeros((n_gt, n_pred), dtype=float), prediction_labels

    # Cellpose normally returns consecutive labels.  Relabelling here keeps the calculation correct even if a future mask contains gaps.
    label_lookup = np.zeros(int(prediction_labels.max()) + 1, dtype=np.int32)
    label_lookup[prediction_labels] = np.arange(1, n_pred + 1)
    prediction_dense = label_lookup[prediction]
    gt_areas = np.bincount(ground_truth.ravel(), minlength=n_gt + 1)
    pred_areas = np.bincount(prediction_dense.ravel(), minlength=n_pred + 1)
    pair_codes = ground_truth.astype(np.int64) * (n_pred + 1) + prediction_dense.astype(np.int64)
    pair_counts = np.bincount(pair_codes.ravel(), minlength=(n_gt + 1) * (n_pred + 1))
    intersections = pair_counts.reshape(n_gt + 1, n_pred + 1)[1:, 1:]
    unions = gt_areas[1:, None] + pred_areas[None, 1:] - intersections
    iou_matrix = np.divide(intersections, unions, out=np.zeros_like(intersections, dtype=float), where=unions > 0)

    return iou_matrix, prediction_labels

def score_prediction(ground_truth, prediction, iou_threshold):
    """Match objects one-to-one and calculate segmentation metrics."""

    n_gt = int(ground_truth.max())
    iou_matrix, prediction_labels = calculate_iou_matrix(ground_truth, prediction)
    n_pred = len(prediction_labels)
    matched_ious = np.array([], dtype=float)

    if n_gt > 0 and n_pred > 0:
        row_indices, column_indices = linear_sum_assignment(-iou_matrix)
        assigned_ious = iou_matrix[row_indices, column_indices]
        matched_ious = assigned_ious[assigned_ious >= iou_threshold]
    true_positives = len(matched_ious)
    false_positives = n_pred - true_positives
    false_negatives = n_gt - true_positives
    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    precision = true_positives / precision_denominator if precision_denominator > 0 else 0.0
    recall = true_positives / recall_denominator if recall_denominator > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {
        "ground_truth_cells": n_gt,
        "predicted_cells": n_pred,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched_iou_sum": float(matched_ious.sum()),
        "mean_matched_iou": (float(matched_ious.mean()) if len(matched_ious) > 0 else np.nan),
    }

def make_config(diameter, cellprob_threshold, flow_threshold, min_size):
    """Create a parameter dictionary with a stable readable ID."""

    diameter_name = "native" if diameter is None else str(diameter)
    config_id = (
        f"diameter_{diameter_name}"
        f"__cellprob_{cellprob_threshold:+.1f}"
        f"__flow_{flow_threshold:.1f}"
        f"__minsize_{min_size}"
    )

    return {
        "config_id": config_id,
        "diameter": diameter,
        "cellprob_threshold": cellprob_threshold,
        "flow_threshold": flow_threshold,
        "min_size": min_size,
    }

def evaluate_config(model, dataset, config, split_name, save_masks=False, keep_predictions=False):
    """Run one configuration and return image- and split-level results."""

    image_results = []
    predictions = {}
    description = f"{split_name}: {config['config_id']}"

    for item in tqdm(dataset, desc=description, leave=False):
        masks, flows, styles = model.eval(
            item["image"],
            channel_axis=None,
            normalize=True,
            diameter=config["diameter"],
            cellprob_threshold=config["cellprob_threshold"],
            flow_threshold=config["flow_threshold"],
            min_size=config["min_size"],
        )

        prediction = masks.astype(np.uint32)
        metrics = score_prediction(item["ground_truth"], prediction, IOU_THRESHOLD)

        image_results.append(
            {"split": split_name, "image": item["filename"], "image_id": item["image_id"], **config, **metrics}
        )

        if save_masks:
            mask_path = MASK_DIR / f"{Path(item['filename']).stem}_optimized_masks.tif"
            tifffile.imwrite(mask_path, prediction)

        if keep_predictions:
            predictions[item["filename"]] = prediction

    results_df = pd.DataFrame(image_results)
    total_tp = int(results_df["true_positives"].sum())
    total_fp = int(results_df["false_positives"].sum())
    total_fn = int(results_df["false_negatives"].sum())
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    total_matched_iou = results_df["matched_iou_sum"].sum()
    summary = {
        "split": split_name,
        **config,
        "images": len(results_df),
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_iou": (total_matched_iou / total_tp if total_tp > 0 else np.nan),
    }

    return results_df, summary, predictions

def select_best(summary_df):
    """Select by F1, using matched IoU and recall as tie-breakers."""

    ranked = summary_df.sort_values(["f1", "mean_matched_iou", "recall"], ascending=[False, False, False])

    return ranked.iloc[0]

def dataframe_row_to_config(row):
    """Convert a results row back to Cellpose parameters."""

    diameter = row["diameter"]

    if pd.isna(diameter):
        diameter = None
    else:
        diameter = int(diameter)

    return make_config(
        diameter=diameter,
        cellprob_threshold=float(row["cellprob_threshold"]),
        flow_threshold=float(row["flow_threshold"]),
        min_size=int(row["min_size"]),
    )

def add_boundaries(axis, label_image, colour_map):
    """Overlay instance boundaries on a Matplotlib axis."""

    boundaries = find_boundaries(label_image, mode="outer")
    overlay = np.ma.masked_where(~boundaries, boundaries)
    axis.imshow(overlay, cmap=colour_map, alpha=0.9)

# ---------------------------------------------------------------------
# Load annotations and create a reproducible split
# ---------------------------------------------------------------------

coco = COCO(str(ANNOTATION_FILE))
image_ids = np.array(sorted(coco.imgs.keys()))
rng = np.random.default_rng(RANDOM_SEED)
rng.shuffle(image_ids)

if MAX_IMAGES is not None:
    image_ids = image_ids[:MAX_IMAGES]

if len(image_ids) < 4:
    raise ValueError("At least four annotated images are required for tuning/validation.")

n_tuning = int(round(len(image_ids) * TUNING_FRACTION))
n_tuning = min(max(n_tuning, 1), len(image_ids) - 1)
tuning_ids = image_ids[:n_tuning]
validation_ids = image_ids[n_tuning:]

print(f"Images selected:   {len(image_ids)}")
print(f"Tuning images:     {len(tuning_ids)}")
print(f"Validation images: {len(validation_ids)}")

# ---------------------------------------------------------------------
# Load images and ground truth once
# ---------------------------------------------------------------------

def load_dataset(selected_ids):
    """Load raw images and matching COCO labels into memory."""

    dataset = []

    for image_id in tqdm(selected_ids, desc="Loading images"):
        metadata = coco.imgs[int(image_id)]
        filename = metadata["file_name"]
        image_path = IMAGE_DIR / filename

        if not image_path.exists():
            print(f"WARNING: image missing: {filename}")
            continue

        image = tifffile.imread(image_path)
        annotation_ids = coco.getAnnIds(imgIds=[int(image_id)])
        annotations = coco.loadAnns(annotation_ids)
        ground_truth = create_ground_truth(coco, annotations, image.shape)

        dataset.append({"image_id": int(image_id), "filename": filename, "image": image, "ground_truth": ground_truth})

    return dataset

tuning_data = load_dataset(tuning_ids)
validation_data = load_dataset(validation_ids)

if len(tuning_data) == 0 or len(validation_data) == 0:
    raise RuntimeError("The tuning or validation split contains no readable images.")

# ---------------------------------------------------------------------
# Load Cellpose
# ---------------------------------------------------------------------

model = models.CellposeModel(gpu=USE_GPU, pretrained_model=MODEL_NAME)

print(f"Cellpose model loaded: {MODEL_NAME}")

# ---------------------------------------------------------------------
# Stage 1: optimise cell-probability and flow thresholds
# ---------------------------------------------------------------------

stage_1_configs = [
    make_config(
        diameter=None,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
        min_size=BASELINE_PARAMETERS["min_size"],
    )
    for cellprob_threshold, flow_threshold in product(CELLPROB_THRESHOLDS, FLOW_THRESHOLDS)
]

all_tuning_results = []
tuning_summaries = []

print("\nStage 1: searching cell-probability and flow thresholds")

for config in stage_1_configs:
    results, summary, _ = evaluate_config(model, tuning_data, config, split_name="tuning")

    all_tuning_results.append(results)
    tuning_summaries.append(summary)

stage_1_summary_df = pd.DataFrame(tuning_summaries)
best_stage_1 = select_best(stage_1_summary_df)

print("\nBest stage-1 configuration:")
print(best_stage_1[["config_id", "precision", "recall", "f1", "mean_matched_iou"]].to_string())

# ---------------------------------------------------------------------
# Stage 2: optimise cell diameter
# ---------------------------------------------------------------------

stage_2_configs = [
    make_config(
        diameter=diameter,
        cellprob_threshold=float(best_stage_1["cellprob_threshold"]),
        flow_threshold=float(best_stage_1["flow_threshold"]),
        min_size=BASELINE_PARAMETERS["min_size"],
    )
    for diameter in DIAMETERS
]

print("\nStage 2: searching cell diameter")

for config in stage_2_configs:
    results, summary, _ = evaluate_config(model, tuning_data, config, split_name="tuning")

    all_tuning_results.append(results)
    tuning_summaries.append(summary)

tuning_results_df = pd.concat(all_tuning_results, ignore_index=True)
tuning_summary_df = pd.DataFrame(tuning_summaries)
best_tuning_row = select_best(tuning_summary_df)
best_config = dataframe_row_to_config(best_tuning_row)

# ---------------------------------------------------------------------
# Save tuning results
# ---------------------------------------------------------------------

tuning_results_path = OUTPUT_DIR / "tuning_per_image.csv"
tuning_summary_path = OUTPUT_DIR / "tuning_summary.csv"

tuning_results_df.to_csv(tuning_results_path, index=False)

tuning_summary_df.sort_values("f1", ascending=False).to_csv(tuning_summary_path, index=False)

print("\nBest overall tuning configuration:")
print(best_tuning_row[["config_id", "precision", "recall", "f1", "mean_matched_iou"]].to_string())

# ---------------------------------------------------------------------
# Held-out validation: baseline versus selected configuration
# ---------------------------------------------------------------------

baseline_config = make_config(**BASELINE_PARAMETERS)

baseline_results, baseline_summary, baseline_predictions = evaluate_config(
    model, validation_data, baseline_config, split_name="validation_baseline", keep_predictions=True
)

optimized_results, optimized_summary, optimized_predictions = evaluate_config(
    model, validation_data, best_config, split_name="validation_optimized", save_masks=True, keep_predictions=True
)

validation_results_df = pd.concat([baseline_results, optimized_results], ignore_index=True)
validation_summary_df = pd.DataFrame([baseline_summary, optimized_summary])
validation_results_path = OUTPUT_DIR / "validation_per_image.csv"
validation_summary_path = OUTPUT_DIR / "validation_summary.csv"

validation_results_df.to_csv(validation_results_path, index=False)
validation_summary_df.to_csv(validation_summary_path, index=False)

# ---------------------------------------------------------------------
# Save selected parameters and experiment details
# ---------------------------------------------------------------------

parameter_report = {
    "model": MODEL_NAME,
    "iou_threshold": IOU_THRESHOLD,
    "random_seed": RANDOM_SEED,
    "tuning_image_ids": [int(value) for value in tuning_ids],
    "validation_image_ids": [int(value) for value in validation_ids],
    "selection_metric": "micro-averaged F1 on tuning images",
    "search_strategy": ("two-stage coordinate search: thresholds, then diameter"),
    "baseline_parameters": BASELINE_PARAMETERS,
    "selected_parameters": {
        "diameter": best_config["diameter"],
        "cellprob_threshold": best_config["cellprob_threshold"],
        "flow_threshold": best_config["flow_threshold"],
        "min_size": best_config["min_size"],
    },
    "tuning_f1": float(best_tuning_row["f1"]),
    "validation_baseline_f1": float(baseline_summary["f1"]),
    "validation_optimized_f1": float(optimized_summary["f1"]),
}

parameter_path = OUTPUT_DIR / "best_parameters.json"

with parameter_path.open("w", encoding="utf-8") as output_file:
    json.dump(parameter_report, output_file, indent=4)

# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

print("\n")
print("=" * 72)
print("CELLPOSE OPTIMISATION RESULTS")
print("=" * 72)

print("\nSelected parameters:")
print(f"Diameter:               {best_config['diameter']}")
print("Cell-probability threshold: " f"{best_config['cellprob_threshold']:.1f}")
print(f"Flow threshold:         {best_config['flow_threshold']:.1f}")
print(f"Minimum object size:    {best_config['min_size']} pixels")

print("\nHeld-out validation performance:")
print(f"Baseline F1:            {baseline_summary['f1']:.3f}")
print(f"Optimised F1:           {optimized_summary['f1']:.3f}")
print("F1 difference:          " f"{optimized_summary['f1'] - baseline_summary['f1']:+.3f}")
print(f"Baseline precision:     {baseline_summary['precision']:.3f}")
print(f"Optimised precision:    {optimized_summary['precision']:.3f}")
print(f"Baseline recall:        {baseline_summary['recall']:.3f}")
print(f"Optimised recall:       {optimized_summary['recall']:.3f}")

if optimized_summary["f1"] > baseline_summary["f1"]:
    print("\nThe selected configuration improved held-out F1.")
elif optimized_summary["f1"] == baseline_summary["f1"]:
    print("\nThe selected configuration tied the baseline on held-out F1.")
else:
    print(
        "\nThe tuning gain did not generalise to the validation images. "
        "Keep the baseline and expand the experiment before deployment."
    )

print(f"\nSaved tuning summary to:\n{tuning_summary_path}")
print(f"\nSaved validation summary to:\n{validation_summary_path}")
print(f"\nSaved selected parameters to:\n{parameter_path}")
print(f"\nSaved optimised validation masks to:\n{MASK_DIR}")

# ---------------------------------------------------------------------
# Plot 1: parameter search
# ---------------------------------------------------------------------

plot_data = tuning_summary_df.sort_values("f1")

plt.figure(figsize=(10, 7))

colours = [
    "tab:orange" if config_id == best_config["config_id"] else "tab:blue" for config_id in plot_data["config_id"]
]

plt.barh(plot_data["config_id"], plot_data["f1"], color=colours)

plt.xlabel("Micro-averaged F1 on tuning images")
plt.ylabel("Cellpose configuration")
plt.title("A172 Cellpose parameter search")
plt.xlim(0, 1)
plt.tight_layout()

parameter_figure_path = OUTPUT_DIR / "parameter_search.png"

plt.savefig(parameter_figure_path, dpi=200, bbox_inches="tight")

plt.show()

# ---------------------------------------------------------------------
# Plot 2: held-out baseline versus optimised performance
# ---------------------------------------------------------------------

comparison_metrics = ["precision", "recall", "f1", "mean_matched_iou"]
baseline_values = [baseline_summary[metric] for metric in comparison_metrics]
optimized_values = [optimized_summary[metric] for metric in comparison_metrics]
x_positions = np.arange(len(comparison_metrics))
bar_width = 0.36

plt.figure(figsize=(9, 6))

plt.bar(x_positions - bar_width / 2, baseline_values, width=bar_width, label="Baseline")

plt.bar(x_positions + bar_width / 2, optimized_values, width=bar_width, label="Optimised")

plt.xticks(x_positions, ["Precision", "Recall", "F1", "Mean matched IoU"])

plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Held-out validation performance")
plt.legend()
plt.tight_layout()

comparison_figure_path = OUTPUT_DIR / "validation_comparison.png"

plt.savefig(comparison_figure_path, dpi=200, bbox_inches="tight")

plt.show()

# ---------------------------------------------------------------------
# Plot 3: qualitative comparison on the hardest validation image
# ---------------------------------------------------------------------

hardest_row = optimized_results.sort_values("f1").iloc[0]
hardest_filename = hardest_row["image"]
hardest_item = next(item for item in validation_data if item["filename"] == hardest_filename)
baseline_mask = baseline_predictions[hardest_filename]
optimized_mask = optimized_predictions[hardest_filename]

fig, axes = plt.subplots(1, 4, figsize=(20, 6))

axes[0].imshow(hardest_item["image"], cmap="gray")
axes[0].set_title("Raw image")

axes[1].imshow(hardest_item["image"], cmap="gray")
add_boundaries(axes[1], hardest_item["ground_truth"], "Greens")
axes[1].set_title(f"Ground truth\n{int(hardest_item['ground_truth'].max())} cells")

axes[2].imshow(hardest_item["image"], cmap="gray")
add_boundaries(axes[2], baseline_mask, "Reds")
axes[2].set_title(
    "Baseline Cellpose\n"
    f"F1 = {baseline_results.loc[baseline_results['image'] == hardest_filename, 'f1'].iloc[0]:.3f}"
)

axes[3].imshow(hardest_item["image"], cmap="gray")
add_boundaries(axes[3], optimized_mask, "Blues")
axes[3].set_title("Optimised Cellpose\n" f"F1 = {hardest_row['f1']:.3f}")

for axis in axes:
    axis.axis("off")

fig.suptitle(f"Hardest held-out image: {hardest_filename}")
plt.tight_layout()

example_figure_path = OUTPUT_DIR / "hardest_validation_example.png"

plt.savefig(example_figure_path, dpi=200, bbox_inches="tight")

plt.show()
