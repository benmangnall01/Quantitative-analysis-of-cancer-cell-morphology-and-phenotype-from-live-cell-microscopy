"""Run the validated Cellpose configuration across all A172 images.

This is the bridge between segmentation optimisation and phenotype analysis.
It reads the frozen parameters selected by 05_optimize_cellpose.py, segments
every A172 image listed in the COCO annotation file, caches label masks, and
creates one table per cell plus one table per image.

Ground-truth masks are deliberately not used for feature extraction here.
The COCO file is used only as a reliable list of A172 image filenames.
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
from scipy.spatial import cKDTree
from skimage.measure import regionprops_table
from skimage.segmentation import find_boundaries
from tqdm import tqdm

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path().resolve().parent
IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "images" / "livecell_train_val_images"
ANNOTATION_FILE = PROJECT_ROOT / "data" / "annotations" / "a172_train.json"
PARAMETER_FILE = PROJECT_ROOT / "outputs" / "cellpose_optimization" / "best_parameters.json"
MASK_DIR = PROJECT_ROOT / "data" / "processed" / "a172_optimized_masks"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "a172_large_scale_analysis"
MASK_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# Set to an integer for a short test run.  None processes every A172 image listed in the COCO annotation file.
MAX_IMAGES = None

# Existing masks are reused only from MASK_DIR, which is dedicated to the optimised parameters from this pipeline.
REUSE_EXISTING_MASKS = True
USE_GPU = True
NEIGHBOUR_RADIUS = 50

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def load_selected_parameters(parameter_file):
    """Load the parameters selected on the held-out validation workflow."""

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
    """Extract well, location, time, and crop from a LIVECell filename."""

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

def add_local_crowding_features(cells):
    """Add neighbour count and nearest-neighbour distance to each cell."""

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
    """Measure morphology, intensity, and local crowding for each cell."""

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

    # image_std is a scalar.  Using np.divide(..., where=...) with a pandas Series and a scalar boolean can fail in some NumPy/Pandas versions, so
    # handle the zero-standard-deviation case explicitly.
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
    """Create image-level summaries without filtering predicted cells."""

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
        "equivalent_diameter_area",
        "circularity",
        "eccentricity",
        "solidity",
        "aspect_ratio",
        "intensity_mean",
        "neighbours_within_50px",
        "nearest_neighbour_distance",
    ]

    for feature in summary_features:
        summary[f"median_{feature}"] = cells[feature].median() if feature in cells and len(cells) > 0 else np.nan

        summary[f"mean_{feature}"] = cells[feature].mean() if feature in cells and len(cells) > 0 else np.nan

    summary["border_touching_cells"] = (
        int(cells["touches_image_border"].sum()) if "touches_image_border" in cells else 0
    )

    summary["border_touching_fraction"] = summary["border_touching_cells"] / len(cells) if len(cells) > 0 else np.nan

    return summary

def add_boundaries(axis, masks, colour_map):
    """Overlay Cellpose instance boundaries on a grayscale image."""

    boundaries = find_boundaries(masks, mode="outer")
    overlay = np.ma.masked_where(~boundaries, boundaries)
    axis.imshow(overlay, cmap=colour_map, alpha=0.9)

# ---------------------------------------------------------------------
# Load selected Cellpose parameters
# ---------------------------------------------------------------------

parameter_report, SELECTED_PARAMETERS = load_selected_parameters(PARAMETER_FILE)

MODEL_NAME = parameter_report.get("model", "cpsam_v2")

print("Selected Cellpose parameters:")
for name, value in SELECTED_PARAMETERS.items():
    print(f"{name}: {value}")

# ---------------------------------------------------------------------
# Get all A172 filenames from COCO
# ---------------------------------------------------------------------

coco = COCO(str(ANNOTATION_FILE))
image_records = sorted(coco.imgs.values(), key=lambda metadata: metadata["file_name"])

if MAX_IMAGES is not None:
    image_records = image_records[:MAX_IMAGES]

if len(image_records) == 0:
    raise RuntimeError("No A172 images were found in the annotation file.")

print(f"\nImages to analyse: {len(image_records)}")

# ---------------------------------------------------------------------
# Segment images and extract single-cell features
# ---------------------------------------------------------------------

model = None
all_cells = []
image_summaries = []
processing_log = []
first_successful_image = None

for image_metadata in tqdm(image_records, desc="Large-scale A172 analysis"):
    image_id = int(image_metadata["id"])
    filename = image_metadata["file_name"]
    image_path = IMAGE_DIR / filename
    mask_path = MASK_DIR / f"{Path(filename).stem}_optimized_masks.tif"
    metadata = {"image": filename, "image_id": image_id, **parse_livecell_filename(filename)}

    if not image_path.exists():
        processing_log.append(
            {**metadata, "status": "missing_image", "mask_source": np.nan, "message": f"Image not found: {image_path}"}
        )
        continue

    try:
        image = tifffile.imread(image_path)
        mask_source = "new_segmentation"

        if REUSE_EXISTING_MASKS and mask_path.exists():
            masks = tifffile.imread(mask_path).astype(np.uint32)

            if masks.shape == image.shape:
                mask_source = "cached_mask"

            else:
                print("\nWARNING: cached mask shape does not match image. " f"Regenerating {filename}.")
                masks = None

        else:
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

        cells = extract_cell_features(image, masks, metadata, SELECTED_PARAMETERS, MODEL_NAME)
        image_summary = create_image_summary(
            image, masks, cells, metadata, SELECTED_PARAMETERS, MODEL_NAME, mask_source
        )

        all_cells.append(cells)
        image_summaries.append(image_summary)

        processing_log.append({**metadata, "status": "completed", "mask_source": mask_source, "message": np.nan})

        if first_successful_image is None:
            first_successful_image = {"image": image, "masks": masks, "filename": filename}

    except Exception as error:
        processing_log.append({**metadata, "status": "failed", "mask_source": np.nan, "message": str(error)})

        print(f"\nWARNING: failed to process {filename}: {error}")

# ---------------------------------------------------------------------
# Combine and save analysis tables
# ---------------------------------------------------------------------

if len(image_summaries) == 0:
    raise RuntimeError("No images were processed successfully.")

if len(all_cells) > 0:
    cells_df = pd.concat(all_cells, ignore_index=True)
else:
    cells_df = pd.DataFrame()
image_summary_df = pd.DataFrame(image_summaries)
processing_log_df = pd.DataFrame(processing_log)
cell_output_path = OUTPUT_DIR / "a172_single_cell_features.csv"
image_output_path = OUTPUT_DIR / "a172_image_summary.csv"
log_output_path = OUTPUT_DIR / "a172_processing_log.csv"
manifest_path = OUTPUT_DIR / "analysis_manifest.json"

cells_df.to_csv(cell_output_path, index=False)

image_summary_df.sort_values(["elapsed_hours", "well", "location", "crop", "image"], na_position="last").to_csv(
    image_output_path, index=False
)

processing_log_df.to_csv(log_output_path, index=False)

analysis_manifest = {
    "analysis": "large-scale A172 single-cell feature extraction",
    "parameter_file": str(PARAMETER_FILE),
    "model": MODEL_NAME,
    "selected_parameters": SELECTED_PARAMETERS,
    "total_images_requested": len(image_records),
    "images_completed": len(image_summary_df),
    "images_failed_or_missing": int((processing_log_df["status"] != "completed").sum()),
    "predicted_cells": int(len(cells_df)),
    "ground_truth_used_for_feature_extraction": False,
    "neighbour_radius_pixels": NEIGHBOUR_RADIUS,
}

with manifest_path.open("w", encoding="utf-8") as output_file:
    json.dump(analysis_manifest, output_file, indent=4)

# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

print("\n")
print("=" * 72)
print("LARGE-SCALE A172 SINGLE-CELL ANALYSIS")
print("=" * 72)

print(f"Images requested:          {len(image_records)}")
print(f"Images completed:          {len(image_summary_df)}")
print(f"Predicted cells measured:  {len(cells_df)}")
print("Images failed or missing:  " f"{(processing_log_df['status'] != 'completed').sum()}")

print("\nCell-level summary:")

summary_columns = [
    "area",
    "equivalent_diameter_area",
    "circularity",
    "eccentricity",
    "solidity",
    "aspect_ratio",
    "neighbours_within_50px",
    "nearest_neighbour_distance",
]

if len(cells_df) > 0:
    print(cells_df[summary_columns].describe().round(3))

print(f"\nSaved single-cell features to:\n{cell_output_path}")
print(f"\nSaved image summaries to:\n{image_output_path}")
print(f"\nSaved processing log to:\n{log_output_path}")
print(f"\nSaved analysis manifest to:\n{manifest_path}")
print(f"\nSaved/reused masks in:\n{MASK_DIR}")

# ---------------------------------------------------------------------
# Plot 1: image-level quality-control overview
# ---------------------------------------------------------------------

plot_images = image_summary_df.sort_values(
    ["elapsed_hours", "well", "location", "crop", "image"], na_position="last"
).reset_index(drop=True)

image_numbers = np.arange(1, len(plot_images) + 1)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].scatter(image_numbers, plot_images["predicted_cells"], alpha=0.8)
axes[0, 0].set_title("Predicted cells per image")
axes[0, 0].set_xlabel("Image sequence")
axes[0, 0].set_ylabel("Predicted cells")

axes[0, 1].scatter(image_numbers, plot_images["cell_coverage_fraction"], alpha=0.8)
axes[0, 1].set_title("Predicted cell coverage")
axes[0, 1].set_xlabel("Image sequence")
axes[0, 1].set_ylabel("Fraction of image covered")

axes[1, 0].scatter(image_numbers, plot_images["median_area"], alpha=0.8)
axes[1, 0].set_title("Median cell area per image")
axes[1, 0].set_xlabel("Image sequence")
axes[1, 0].set_ylabel("Median area (pixels)")

axes[1, 1].scatter(image_numbers, plot_images["median_nearest_neighbour_distance"], alpha=0.8)
axes[1, 1].set_title("Median nearest-neighbour distance")
axes[1, 1].set_xlabel("Image sequence")
axes[1, 1].set_ylabel("Distance (pixels)")

fig.suptitle("A172 segmentation and measurement quality control")
plt.tight_layout()

qc_figure_path = OUTPUT_DIR / "a172_image_level_qc.png"

plt.savefig(qc_figure_path, dpi=200, bbox_inches="tight")

# ---------------------------------------------------------------------
# Plot 2: cell morphology overview
# ---------------------------------------------------------------------

if len(cells_df) > 0:
    max_plot_cells = 5000

    if len(cells_df) > max_plot_cells:
        plot_cells = cells_df.sample(max_plot_cells, random_state=42)
    else:
        plot_cells = cells_df

    plt.figure(figsize=(8, 6))

    plt.scatter(plot_cells["area"], plot_cells["circularity"], alpha=0.25, s=12)

    plt.xlabel("Cell area (pixels)")
    plt.ylabel("Circularity")
    plt.title("A172 predicted-cell morphology overview")
    plt.tight_layout()

    morphology_figure_path = OUTPUT_DIR / "a172_area_vs_circularity.png"

    plt.savefig(morphology_figure_path, dpi=200, bbox_inches="tight")

# ---------------------------------------------------------------------
# Plot 3: representative segmentation quality-control overlay
# ---------------------------------------------------------------------

if first_successful_image is not None:
    plt.figure(figsize=(10, 7))
    plt.imshow(first_successful_image["image"], cmap="gray")
    add_boundaries(plt.gca(), first_successful_image["masks"], "Reds")

    plt.title("Representative large-scale Cellpose segmentation\n" f"{first_successful_image['filename']}")
    plt.axis("off")
    plt.tight_layout()

    overlay_figure_path = OUTPUT_DIR / "a172_representative_overlay.png"

    plt.savefig(overlay_figure_path, dpi=200, bbox_inches="tight")