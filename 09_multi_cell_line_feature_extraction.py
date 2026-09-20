"""Extract matched single-cell features from several LIVECell cell lines.

This script is the bridge from validated A172 phenotype analysis to a
multi-cell-line classifier.  It applies the A172-selected Cellpose parameters
to a configurable panel of LIVECell cell lines, measures the same morphology
and crowding features for every predicted cell, and saves transparent
cell-line-specific quality-control and segmentation-evaluation summaries.

The A172-tuned parameters are transferred rather than re-optimised.  Optional
COCO annotation files allow that transfer to be quantified per cell line, but
their absence does not prevent feature extraction.
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from cellpose import models
from pycocotools.coco import COCO
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from skimage.measure import regionprops_table
from tqdm import tqdm

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path().resolve().parent
IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "images" / "livecell_train_val_images"
ANNOTATION_DIR = PROJECT_ROOT / "data" / "annotations"
PARAMETER_FILE = PROJECT_ROOT / "outputs" / "cellpose_optimization" / "best_parameters.json"

# A separate directory prevents accidental mixing of multi-line masks with the A172-only masks produced by 06_large_scale_a172_analysis.py.
MASK_DIR = PROJECT_ROOT / "data" / "processed" / "multi_cell_line_masks"
A172_MASK_DIR = PROJECT_ROOT / "data" / "processed" / "a172_optimized_masks"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "multi_cell_line_analysis"
MASK_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# This five-line panel gives the later classifier biologically and morphologically diverse classes while keeping the first full run tractable.
# Add "SHSY5Y", "SKOV3", or "SkBr3" after this run if those images are present.
CELL_LINES = ["A172", "BT474", "BV2", "Huh7", "MCF7"]

# Images are evenly spaced through the sorted available filenames so that one short run covers the range of recorded times rather than only early images.
# Set to None only when you are ready to process every available image.
MAX_IMAGES_PER_CELL_LINE = 50
REUSE_EXISTING_MASKS = True
REUSE_A172_MASKS = True
USE_GPU = True
NEIGHBOUR_RADIUS = 50

# Optional per-line transfer evaluation.  The script looks for annotation files named e.g. data/annotations/bt474_train.json.  Missing files are
# reported, not treated as errors.
EVALUATE_WITH_ANNOTATIONS = True
MAX_EVALUATION_IMAGES_PER_CELL_LINE = 20
IOU_THRESHOLD = 0.50

# These conservative per-line bounds remove extreme predicted-object sizes while retaining genuine cell-line size differences for the classifier.
LOWER_AREA_QUANTILE = 0.005
UPPER_AREA_QUANTILE = 0.995

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def load_selected_parameters(parameter_file):
    """Load the held-out A172 Cellpose configuration from script 05."""

    if not parameter_file.exists():
        raise FileNotFoundError(
            "Optimisation parameters were not found. Run "
            "05_optimize_cellpose.py first.\n"
            f"Expected file: {parameter_file}"
        )

    with parameter_file.open("r", encoding="utf-8") as input_file:
        parameter_report = json.load(input_file)
    selected_parameters = parameter_report.get("selected_parameters")

    if selected_parameters is None:
        raise ValueError("The parameter file does not contain 'selected_parameters'.")

    required_parameters = {"diameter", "cellprob_threshold", "flow_threshold", "min_size"}
    missing_parameters = required_parameters - set(selected_parameters)

    if missing_parameters:
        raise ValueError("The parameter file is missing: " f"{sorted(missing_parameters)}")

    return parameter_report, selected_parameters

def parse_livecell_filename(filename):
    """Extract cell-line, field, time, and crop from a LIVECell filename."""

    pattern = re.compile(
        r"^(?P<cell_line>[^_]+)_"
        r"(?P<modality>[^_]+)_"
        r"(?P<well>[A-Za-z]\d+)_"
        r"(?P<location>\d+)_"
        r"(?P<elapsed>\d+d\d+h\d+m)_"
        r"(?P<crop>\d+)\.tif$"
    )

    match = pattern.match(filename)

    if match is None:
        return {
            "cell_line": np.nan,
            "modality": np.nan,
            "well": np.nan,
            "location": np.nan,
            "elapsed_time": np.nan,
            "elapsed_hours": np.nan,
            "crop": np.nan,
        }

    values = match.groupdict()
    elapsed_match = re.fullmatch(r"(?P<days>\d+)d(?P<hours>\d+)h(?P<minutes>\d+)m", values["elapsed"])
    elapsed_hours = 24 * int(elapsed_match["days"]) + int(elapsed_match["hours"]) + int(elapsed_match["minutes"]) / 60

    return {
        "cell_line": values["cell_line"],
        "modality": values["modality"],
        "well": values["well"],
        "location": int(values["location"]),
        "elapsed_time": values["elapsed"],
        "elapsed_hours": elapsed_hours,
        "crop": int(values["crop"]),
    }

def select_image_paths(cell_line):
    """Choose deterministic, time-spread image paths for one cell line."""

    all_paths = sorted(IMAGE_DIR.glob(f"{cell_line}_*.tif"))

    if len(all_paths) == 0:
        return []

    if MAX_IMAGES_PER_CELL_LINE is None or len(all_paths) <= MAX_IMAGES_PER_CELL_LINE:
        return all_paths

    selected_indices = np.linspace(0, len(all_paths) - 1, num=MAX_IMAGES_PER_CELL_LINE, dtype=int)

    return [all_paths[index] for index in np.unique(selected_indices)]

def find_annotation_file(cell_line):
    """Return a per-line annotation file when the user has provided one."""

    candidate_paths = [ANNOTATION_DIR / f"{cell_line.lower()}_train.json", ANNOTATION_DIR / f"{cell_line}_train.json"]

    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return candidate_path

    return None

def create_ground_truth(coco, annotations, image_shape):
    """Convert the COCO polygons to a single instance-label image."""

    ground_truth = np.zeros(image_shape, dtype=np.uint32)

    for label, annotation in enumerate(annotations, start=1):
        ground_truth[coco.annToMask(annotation) > 0] = label

    return ground_truth

def calculate_iou_matrix(ground_truth, prediction):
    """Calculate all ground-truth/prediction instance IoUs efficiently."""

    n_ground_truth = int(ground_truth.max())
    prediction_labels = np.unique(prediction)
    prediction_labels = prediction_labels[prediction_labels != 0]
    n_prediction = len(prediction_labels)

    if n_ground_truth == 0 or n_prediction == 0:
        return (np.zeros((n_ground_truth, n_prediction), dtype=float), prediction_labels)

    label_lookup = np.zeros(int(prediction_labels.max()) + 1, dtype=np.int32)
    label_lookup[prediction_labels] = np.arange(1, n_prediction + 1)
    prediction_dense = label_lookup[prediction]
    ground_truth_areas = np.bincount(ground_truth.ravel(), minlength=n_ground_truth + 1)
    prediction_areas = np.bincount(prediction_dense.ravel(), minlength=n_prediction + 1)
    pair_codes = ground_truth.astype(np.int64) * (n_prediction + 1) + prediction_dense.astype(np.int64)
    pair_counts = np.bincount(pair_codes.ravel(), minlength=(n_ground_truth + 1) * (n_prediction + 1))
    intersections = pair_counts.reshape(n_ground_truth + 1, n_prediction + 1)[1:, 1:]
    unions = ground_truth_areas[1:, None] + prediction_areas[None, 1:] - intersections
    iou_matrix = np.divide(intersections, unions, out=np.zeros_like(intersections, dtype=float), where=unions > 0)

    return iou_matrix, prediction_labels

def score_prediction(ground_truth, prediction):
    """Calculate one-to-one instance segmentation metrics at the set IoU."""

    n_ground_truth = int(ground_truth.max())
    iou_matrix, prediction_labels = calculate_iou_matrix(ground_truth, prediction)
    n_prediction = len(prediction_labels)
    matched_ious = np.array([], dtype=float)

    if n_ground_truth > 0 and n_prediction > 0:
        row_indices, column_indices = linear_sum_assignment(-iou_matrix)
        assigned_ious = iou_matrix[row_indices, column_indices]
        matched_ious = assigned_ious[assigned_ious >= IOU_THRESHOLD]
    true_positives = len(matched_ious)
    false_positives = n_prediction - true_positives
    false_negatives = n_ground_truth - true_positives
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {
        "ground_truth_cells": n_ground_truth,
        "predicted_cells": n_prediction,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_iou": (float(matched_ious.mean()) if len(matched_ious) > 0 else np.nan),
    }

def add_local_crowding_features(cells):
    """Add per-cell local crowding measures from centroid positions."""

    cells = cells.copy()

    if len(cells) == 0:
        cells["neighbours_within_50px"] = pd.Series(dtype="int64")
        cells["nearest_neighbour_distance"] = pd.Series(dtype="float64")
        return cells

    coordinates = cells[["centroid_row", "centroid_col"]].to_numpy()
    tree = cKDTree(coordinates)
    neighbour_lists = tree.query_ball_point(coordinates, r=NEIGHBOUR_RADIUS)
    cells["neighbours_within_50px"] = [len(neighbours) - 1 for neighbours in neighbour_lists]

    if len(cells) > 1:
        distances, _ = tree.query(coordinates, k=2)
        cells["nearest_neighbour_distance"] = distances[:, 1]
    else:
        cells["nearest_neighbour_distance"] = np.nan

    return cells

def extract_cell_features(image, masks, metadata, parameters, model_name):
    """Measure the morphology, intensity, and crowding of predicted cells."""

    properties = [
        "label",
        "area",
        "perimeter",
        "perimeter_crofton",
        "eccentricity",
        "solidity",
        "extent",
        "equivalent_diameter_area",
        "major_axis_length",
        "minor_axis_length",
        "orientation",
        "centroid",
        "bbox",
        "intensity_mean",
        "intensity_median",
        "intensity_std",
        "intensity_min",
        "intensity_max",
    ]

    measurements = regionprops_table(masks, intensity_image=image, properties=properties)
    cells = pd.DataFrame(measurements)

    if len(cells) == 0:
        return cells

    cells = cells.rename(
        columns={
            "label": "cell_id",
            "centroid-0": "centroid_row",
            "centroid-1": "centroid_col",
            "bbox-0": "bbox_min_row",
            "bbox-1": "bbox_min_col",
            "bbox-2": "bbox_max_row",
            "bbox-3": "bbox_max_col",
        }
    )

    image_height, image_width = image.shape
    image_mean = float(np.mean(image))
    image_std = float(np.std(image))

    cells["aspect_ratio"] = np.divide(
        cells["major_axis_length"],
        cells["minor_axis_length"],
        out=np.full(len(cells), np.nan),
        where=cells["minor_axis_length"] > 0,
    )

    cells["circularity"] = np.divide(
        4 * np.pi * cells["area"],
        cells["perimeter_crofton"] ** 2,
        out=np.full(len(cells), np.nan),
        where=cells["perimeter_crofton"] > 0,
    )

    cells["centroid_to_edge_distance"] = np.minimum.reduce(
        [
            cells["centroid_row"],
            cells["centroid_col"],
            image_height - 1 - cells["centroid_row"],
            image_width - 1 - cells["centroid_col"],
        ]
    )

    cells["touches_image_border"] = (
        (cells["bbox_min_row"] == 0)
        | (cells["bbox_min_col"] == 0)
        | (cells["bbox_max_row"] == image_height)
        | (cells["bbox_max_col"] == image_width)
    )

    cells["image_intensity_mean"] = image_mean
    cells["image_intensity_std"] = image_std

    if image_std > 0:
        cells["cell_mean_intensity_zscore"] = (cells["intensity_mean"] - image_mean) / image_std
    else:
        cells["cell_mean_intensity_zscore"] = np.nan

    cells = add_local_crowding_features(cells)

    for column, value in metadata.items():
        cells[column] = value

    cells["model_name"] = model_name

    for column, value in parameters.items():
        cells[f"cellpose_{column}"] = value

    return cells

def create_image_summary(image, masks, cells, metadata, parameters, model_name, mask_source):
    """Create unfiltered image-level quality-control summaries."""

    image_area = image.size
    cell_pixels = int(np.count_nonzero(masks))
    summary = {
        **metadata,
        "image_height": image.shape[0],
        "image_width": image.shape[1],
        "image_area_pixels": image_area,
        "predicted_cells": len(cells),
        "cell_coverage_fraction": cell_pixels / image_area,
        "cell_density_per_100k_px": len(cells) / image_area * 100_000,
        "raw_image_mean": float(np.mean(image)),
        "raw_image_std": float(np.std(image)),
        "mask_source": mask_source,
        "model_name": model_name,
    }

    summary.update({f"cellpose_{column}": value for column, value in parameters.items()})

    summary_features = [
        "area",
        "circularity",
        "solidity",
        "aspect_ratio",
        "neighbours_within_50px",
        "nearest_neighbour_distance",
    ]

    for feature in summary_features:
        summary[f"median_{feature}"] = cells[feature].median() if feature in cells and len(cells) > 0 else np.nan

    summary["border_touching_cells"] = (
        int(cells["touches_image_border"].sum()) if "touches_image_border" in cells else 0
    )
    summary["border_touching_fraction"] = summary["border_touching_cells"] / len(cells) if len(cells) > 0 else np.nan

    return summary

def add_classifier_qc_flags(cells):
    """Flag feature-extraction problems while preserving every raw cell."""

    cells = cells.copy()
    feature_columns = [
        "area",
        "circularity",
        "solidity",
        "aspect_ratio",
        "neighbours_within_50px",
        "nearest_neighbour_distance",
    ]

    border_values = cells["touches_image_border"]
    if pd.api.types.is_bool_dtype(border_values):
        cells["qc_touches_image_border"] = border_values.fillna(False)
    else:
        cells["qc_touches_image_border"] = border_values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})

    cells["qc_extreme_area_within_cell_line"] = False
    area_limit_records = []

    for cell_line, indices in cells.groupby("cell_line").groups.items():
        areas = cells.loc[indices, "area"]
        finite_areas = areas.loc[np.isfinite(areas)]
        lower_limit = finite_areas.quantile(LOWER_AREA_QUANTILE)
        upper_limit = finite_areas.quantile(UPPER_AREA_QUANTILE)

        cells.loc[indices, "qc_extreme_area_within_cell_line"] = (areas < lower_limit) | (areas > upper_limit)

        area_limit_records.append(
            {"cell_line": cell_line, "area_qc_lower_pixels": lower_limit, "area_qc_upper_pixels": upper_limit}
        )

    feature_values = cells[feature_columns].to_numpy(dtype=float)
    cells["qc_missing_feature"] = ~np.isfinite(feature_values).all(axis=1)

    cells["qc_include_classifier"] = ~(
        cells["qc_touches_image_border"] | cells["qc_extreme_area_within_cell_line"] | cells["qc_missing_feature"]
    )

    exclusion_reason = pd.Series("", index=cells.index, dtype="object")

    for condition, reason in [
        (cells["qc_touches_image_border"], "touches_image_border"),
        (cells["qc_extreme_area_within_cell_line"], "extreme_area_within_cell_line"),
        (cells["qc_missing_feature"], "missing_feature"),
    ]:
        appended_reason = np.where(exclusion_reason.eq(""), reason, exclusion_reason + ";" + reason)
        exclusion_reason = exclusion_reason.where(~condition, appended_reason)

    cells["qc_exclusion_reason"] = exclusion_reason.replace("", "included")

    return cells, pd.DataFrame(area_limit_records)

def make_cell_line_summary(raw_cells, qc_cells, image_summaries, area_limits):
    """Combine cell counts, QC, morphology, and image-level QC by cell line."""

    raw_summary = (
        raw_cells.groupby("cell_line")
        .agg(
            predicted_cells=("cell_id", "size"),
            images_with_features=("image", "nunique"),
            median_area=("area", "median"),
            median_circularity=("circularity", "median"),
            median_solidity=("solidity", "median"),
            median_aspect_ratio=("aspect_ratio", "median"),
        )
        .reset_index()
    )

    qc_summary = (
        qc_cells.groupby("cell_line")
        .agg(
            classifier_qc_passing_cells=("qc_include_classifier", "sum"),
            border_touching_fraction=("qc_touches_image_border", "mean"),
            extreme_area_fraction=("qc_extreme_area_within_cell_line", "mean"),
            missing_feature_fraction=("qc_missing_feature", "mean"),
        )
        .reset_index()
    )

    image_summary = (
        image_summaries.groupby("cell_line")
        .agg(
            images_completed=("image", "nunique"),
            median_cell_coverage_fraction=("cell_coverage_fraction", "median"),
            median_predicted_cells_per_image=("predicted_cells", "median"),
        )
        .reset_index()
    )

    summary = (
        raw_summary.merge(qc_summary, on="cell_line", how="left")
        .merge(image_summary, on="cell_line", how="left")
        .merge(area_limits, on="cell_line", how="left")
    )

    summary["classifier_qc_passing_fraction"] = summary["classifier_qc_passing_cells"] / summary["predicted_cells"]

    return summary.sort_values("cell_line").reset_index(drop=True)

def make_evaluation_summary(evaluation_records):
    """Summarise optional ground-truth segmentation scoring by cell line."""

    evaluation = pd.DataFrame(evaluation_records)

    if len(evaluation) == 0:
        return evaluation, pd.DataFrame()

    evaluated = evaluation.loc[evaluation["status"] == "evaluated"].copy()

    if len(evaluated) == 0:
        return evaluation, pd.DataFrame()

    summary = (
        evaluated.groupby("cell_line")
        .agg(
            evaluation_images=("image", "nunique"),
            micro_true_positives=("true_positives", "sum"),
            micro_false_positives=("false_positives", "sum"),
            micro_false_negatives=("false_negatives", "sum"),
            mean_matched_iou=("mean_matched_iou", "mean"),
        )
        .reset_index()
    )

    precision_denominator = summary["micro_true_positives"] + summary["micro_false_positives"]
    recall_denominator = summary["micro_true_positives"] + summary["micro_false_negatives"]

    summary["precision"] = np.divide(
        summary["micro_true_positives"],
        precision_denominator,
        out=np.zeros(len(summary), dtype=float),
        where=precision_denominator > 0,
    )
    summary["recall"] = np.divide(
        summary["micro_true_positives"],
        recall_denominator,
        out=np.zeros(len(summary), dtype=float),
        where=recall_denominator > 0,
    )
    summary["f1"] = np.divide(
        2 * summary["precision"] * summary["recall"],
        summary["precision"] + summary["recall"],
        out=np.zeros(len(summary), dtype=float),
        where=(summary["precision"] + summary["recall"]) > 0,
    )

    return evaluation, summary.sort_values("cell_line")

# ---------------------------------------------------------------------
# Select images and load optional annotation indexes
# ---------------------------------------------------------------------

parameter_report, SELECTED_PARAMETERS = load_selected_parameters(PARAMETER_FILE)
MODEL_NAME = parameter_report.get("model", "cpsam_v2")

if not IMAGE_DIR.exists():
    raise FileNotFoundError("LIVECell image directory was not found.\n" f"Expected directory: {IMAGE_DIR}")

selected_records = []

for cell_line in CELL_LINES:
    image_paths = select_image_paths(cell_line)

    if len(image_paths) == 0:
        print(f"WARNING: no images found for {cell_line}; skipping it.")
        continue

    for sequence_index, image_path in enumerate(image_paths, start=1):
        selected_records.append(
            {
                "cell_line": cell_line,
                "image": image_path.name,
                "image_path": str(image_path),
                "selection_sequence": sequence_index,
            }
        )

if len(selected_records) == 0:
    raise RuntimeError("No images were selected. Check IMAGE_DIR and CELL_LINES.")

selected_images_df = pd.DataFrame(selected_records)
annotation_indexes = {}

if EVALUATE_WITH_ANNOTATIONS:
    for cell_line in selected_images_df["cell_line"].unique():
        annotation_file = find_annotation_file(cell_line)

        if annotation_file is None:
            annotation_indexes[cell_line] = None
            continue

        coco = COCO(str(annotation_file))
        filename_to_id = {metadata["file_name"]: image_id for image_id, metadata in coco.imgs.items()}
        annotation_indexes[cell_line] = {
            "coco": coco,
            "filename_to_id": filename_to_id,
            "annotation_file": annotation_file,
        }

print("Selected Cellpose parameters:")
for name, value in SELECTED_PARAMETERS.items():
    print(f"{name}: {value}")

print("\nImages selected by cell line:")
print(selected_images_df.groupby("cell_line").size().to_string())

# ---------------------------------------------------------------------
# Segment selected images and extract matched single-cell features
# ---------------------------------------------------------------------

model = None
all_cells = []
image_summaries = []
processing_log = []
evaluation_records = []
evaluation_counts = {cell_line: 0 for cell_line in selected_images_df["cell_line"].unique()}

for record in tqdm(selected_records, desc="Multi-cell-line feature extraction"):
    selected_cell_line = record["cell_line"]
    filename = record["image"]
    image_path = Path(record["image_path"])
    metadata = {"image": filename, **parse_livecell_filename(filename)}

    # Use the requested line as an explicit class label even if a malformed filename could not be parsed.
    metadata["cell_line"] = selected_cell_line

    mask_path = MASK_DIR / f"{Path(filename).stem}_optimized_masks.tif"
    a172_mask_path = A172_MASK_DIR / f"{Path(filename).stem}_optimized_masks.tif"

    if not image_path.exists():
        processing_log.append(
            {**metadata, "status": "missing_image", "mask_source": np.nan, "message": f"Image not found: {image_path}"}
        )
        continue

    try:
        image = tifffile.imread(image_path)
        masks = None
        mask_source = ""

        if REUSE_EXISTING_MASKS and mask_path.exists():
            masks = tifffile.imread(mask_path).astype(np.uint32)
            mask_source = "multi_line_cached_mask"

        elif REUSE_EXISTING_MASKS and REUSE_A172_MASKS and selected_cell_line == "A172" and a172_mask_path.exists():
            masks = tifffile.imread(a172_mask_path).astype(np.uint32)
            mask_source = "a172_optimised_cached_mask"

        if masks is not None and masks.shape != image.shape:
            print("\nWARNING: cached mask shape does not match image. " f"Regenerating {filename}.")
            masks = None

        if masks is None:
            if model is None:
                model = models.CellposeModel(gpu=USE_GPU, pretrained_model=MODEL_NAME)
                print(f"\nCellpose model loaded: {MODEL_NAME}")

            masks, flows, styles = model.eval(
                image,
                channel_axis=None,
                normalize=True,
                diameter=SELECTED_PARAMETERS["diameter"],
                cellprob_threshold=(SELECTED_PARAMETERS["cellprob_threshold"]),
                flow_threshold=SELECTED_PARAMETERS["flow_threshold"],
                min_size=SELECTED_PARAMETERS["min_size"],
            )
            masks = masks.astype(np.uint32)
            tifffile.imwrite(mask_path, masks)
            mask_source = "new_segmentation"
        cells = extract_cell_features(image, masks, metadata, SELECTED_PARAMETERS, MODEL_NAME)
        image_summary = create_image_summary(
            image, masks, cells, metadata, SELECTED_PARAMETERS, MODEL_NAME, mask_source
        )

        all_cells.append(cells)
        image_summaries.append(image_summary)
        processing_log.append({**metadata, "status": "completed", "mask_source": mask_source, "message": np.nan})

        annotation_index = annotation_indexes.get(selected_cell_line)

        if not EVALUATE_WITH_ANNOTATIONS:
            evaluation_records.append({**metadata, "status": "evaluation_disabled"})

        elif annotation_index is None:
            evaluation_records.append({**metadata, "status": "annotation_file_not_found"})

        elif evaluation_counts[selected_cell_line] >= MAX_EVALUATION_IMAGES_PER_CELL_LINE:
            evaluation_records.append({**metadata, "status": "evaluation_limit_reached"})

        else:
            image_id = annotation_index["filename_to_id"].get(filename)

            if image_id is None:
                evaluation_records.append({**metadata, "status": "not_in_annotation_file"})

            else:
                coco = annotation_index["coco"]
                annotation_ids = coco.getAnnIds(imgIds=[image_id])
                annotations = coco.loadAnns(annotation_ids)
                ground_truth = create_ground_truth(coco, annotations, image.shape)
                scores = score_prediction(ground_truth, masks)

                evaluation_records.append(
                    {
                        **metadata,
                        "status": "evaluated",
                        "annotation_file": str(annotation_index["annotation_file"]),
                        **scores,
                    }
                )
                evaluation_counts[selected_cell_line] += 1

    except Exception as error:
        processing_log.append({**metadata, "status": "failed", "mask_source": np.nan, "message": str(error)})
        print(f"\nWARNING: failed to process {filename}: {error}")

# ---------------------------------------------------------------------
# Combine outputs and apply transparent classifier QC
# ---------------------------------------------------------------------

if len(image_summaries) == 0:
    raise RuntimeError("No images were processed successfully.")

raw_cells_df = pd.concat(all_cells, ignore_index=True)
image_summary_df = pd.DataFrame(image_summaries)
processing_log_df = pd.DataFrame(processing_log)

if len(raw_cells_df) == 0:
    raise RuntimeError("No predicted cells were available for feature extraction.")

qc_cells_df, area_limits_df = add_classifier_qc_flags(raw_cells_df)
classifier_cells_df = qc_cells_df.loc[qc_cells_df["qc_include_classifier"]].copy()
cell_line_summary_df = make_cell_line_summary(raw_cells_df, qc_cells_df, image_summary_df, area_limits_df)

evaluation_df, evaluation_summary_df = make_evaluation_summary(evaluation_records)

# ---------------------------------------------------------------------
# Save tables and manifest
# ---------------------------------------------------------------------

selected_images_output_path = OUTPUT_DIR / "selected_images.csv"
raw_cells_output_path = OUTPUT_DIR / "multicell_line_single_cell_features.csv"
qc_cells_output_path = OUTPUT_DIR / "multicell_line_cells_with_qc.csv"
classifier_cells_output_path = OUTPUT_DIR / "multicell_line_classifier_cells.csv"
image_summary_output_path = OUTPUT_DIR / "multicell_line_image_summary.csv"
processing_log_output_path = OUTPUT_DIR / "multicell_line_processing_log.csv"
evaluation_output_path = OUTPUT_DIR / "multicell_line_segmentation_evaluation.csv"
evaluation_summary_output_path = OUTPUT_DIR / "multicell_line_segmentation_evaluation_summary.csv"
cell_line_summary_output_path = OUTPUT_DIR / "multicell_line_summary.csv"
area_limits_output_path = OUTPUT_DIR / "multicell_line_area_qc_limits.csv"
manifest_output_path = OUTPUT_DIR / "multi_cell_line_analysis_manifest.json"

selected_images_df.to_csv(selected_images_output_path, index=False)
raw_cells_df.to_csv(raw_cells_output_path, index=False)
qc_cells_df.to_csv(qc_cells_output_path, index=False)
classifier_cells_df.to_csv(classifier_cells_output_path, index=False)

image_summary_df.sort_values(
    ["cell_line", "elapsed_hours", "well", "location", "crop", "image"], na_position="last"
).to_csv(image_summary_output_path, index=False)

processing_log_df.to_csv(processing_log_output_path, index=False)
evaluation_df.to_csv(evaluation_output_path, index=False)
evaluation_summary_df.to_csv(evaluation_summary_output_path, index=False)
cell_line_summary_df.to_csv(cell_line_summary_output_path, index=False)
area_limits_df.to_csv(area_limits_output_path, index=False)

manifest = {
    "analysis": "Multi-cell-line LIVECell feature extraction",
    "selected_cell_lines": CELL_LINES,
    "cell_lines_completed": sorted(image_summary_df["cell_line"].unique().tolist()),
    "max_images_per_cell_line": MAX_IMAGES_PER_CELL_LINE,
    "images_selected": int(len(selected_images_df)),
    "images_completed": int(len(image_summary_df)),
    "images_failed_or_missing": int((processing_log_df["status"] != "completed").sum()),
    "predicted_cells": int(len(raw_cells_df)),
    "classifier_qc_passing_cells": int(len(classifier_cells_df)),
    "model": MODEL_NAME,
    "selected_parameters": SELECTED_PARAMETERS,
    "a172_mask_reuse_enabled": REUSE_A172_MASKS,
    "ground_truth_evaluation_enabled": EVALUATE_WITH_ANNOTATIONS,
    "iou_threshold": IOU_THRESHOLD,
    "qc_area_quantiles_within_cell_line": {"lower": LOWER_AREA_QUANTILE, "upper": UPPER_AREA_QUANTILE},
    "classifier_caveat": (
        "These features are suitable inputs for group-aware model evaluation, "
        "but a classifier must split by image or well/location rather than "
        "randomly splitting individual cells."
    ),
}

with manifest_output_path.open("w", encoding="utf-8") as output_file:
    json.dump(manifest, output_file, indent=4)

# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

print("\n")
print("=" * 72)
print("MULTI-CELL-LINE FEATURE EXTRACTION")
print("=" * 72)
print(f"Images selected:            {len(selected_images_df)}")
print(f"Images completed:           {len(image_summary_df)}")
print(f"Predicted cells measured:   {len(raw_cells_df)}")
print(f"Classifier-QC cells:        {len(classifier_cells_df)}")
print("Cell lines completed:       " f"{', '.join(sorted(image_summary_df['cell_line'].unique()))}")

print("\nCell-line feature and QC summary:")
print(cell_line_summary_df.round(3).to_string(index=False))

if len(evaluation_summary_df) > 0:
    print("\nOptional ground-truth segmentation evaluation:")
    print(evaluation_summary_df.round(3).to_string(index=False))
else:
    print(
        "\nNo ground-truth evaluation was completed. To enable it, add "
        "per-line COCO annotation files such as bt474_train.json to "
        "data/annotations."
    )

print(f"\nSaved classifier-ready cells to:\n{classifier_cells_output_path}")
print(f"\nSaved cell-line summary to:\n{cell_line_summary_output_path}")
print(f"\nSaved segmentation evaluation to:\n{evaluation_output_path}")

# ---------------------------------------------------------------------
# Plot: feature-extraction and optional segmentation-QC overview
# ---------------------------------------------------------------------

line_order = cell_line_summary_df["cell_line"].tolist()
line_positions = np.arange(len(line_order))

figure, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].bar(line_positions - 0.18, cell_line_summary_df["predicted_cells"], width=0.36, label="Predicted cells")
axes[0, 0].bar(
    line_positions + 0.18, cell_line_summary_df["classifier_qc_passing_cells"], width=0.36, label="Classifier-QC cells"
)
axes[0, 0].set_xticks(line_positions, line_order, rotation=25)
axes[0, 0].set_ylabel("Cells")
axes[0, 0].set_title("Cells retained for the multi-line classifier")
axes[0, 0].legend()

coverage_data = [
    image_summary_df.loc[image_summary_df["cell_line"] == cell_line, "cell_coverage_fraction"].dropna().to_numpy()
    for cell_line in line_order
]
axes[0, 1].boxplot(coverage_data)
axes[0, 1].set_xticks(np.arange(1, len(line_order) + 1), line_order)
axes[0, 1].set_ylabel("Predicted cell coverage fraction")
axes[0, 1].set_title("Segmentation coverage by cell line")
axes[0, 1].tick_params(axis="x", rotation=25)

axes[1, 0].bar(line_positions, cell_line_summary_df["median_area"], color="tab:purple")
axes[1, 0].set_xticks(line_positions, line_order, rotation=25)
axes[1, 0].set_ylabel("Pixels")
axes[1, 0].set_title("Median predicted cell area by line")

if len(evaluation_summary_df) > 0:
    evaluation_plot_data = evaluation_summary_df.set_index("cell_line").reindex(line_order)
    axes[1, 1].bar(line_positions - 0.18, evaluation_plot_data["f1"], width=0.36, label="F1 at IoU 0.50")
    axes[1, 1].bar(
        line_positions + 0.18, evaluation_plot_data["mean_matched_iou"], width=0.36, label="Mean matched IoU"
    )
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_xticks(line_positions, line_order, rotation=25)
    axes[1, 1].set_ylabel("Score")
    axes[1, 1].set_title("Optional transfer segmentation evaluation")
    axes[1, 1].legend()
else:
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.5,
        0.5,
        "No per-line annotation files found.\n" "Feature extraction completed without ground-truth transfer scores.",
        ha="center",
        va="center",
        wrap=True,
    )

figure.suptitle("Multi-cell-line feature extraction and quality control")
plt.tight_layout()

figure_path = OUTPUT_DIR / "multicell_line_qc_overview.png"
plt.savefig(figure_path, dpi=200, bbox_inches="tight")