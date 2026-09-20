"""Validate exploratory A172 morphology and crowding phenotypes.

This script follows 07_a172_phenotype_exploration.py.  It checks that the
exploratory K-means groups correspond to visibly plausible cells, measures
their stability under cell-level bootstrap resampling, and summarises cluster
composition using images/fields as the time-course unit.

The validation supports reproducible exploratory phenotypes; it does not turn
the clusters into confirmed biological states or prove individual cells change
from one cluster to another.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import RobustScaler

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path().resolve().parent
INPUT_FILE = PROJECT_ROOT / "outputs" / "a172_phenotype_exploration" / "a172_phenotype_analysis_cells.csv"
IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "images" / "livecell_train_val_images"
MASK_DIR = PROJECT_ROOT / "data" / "processed" / "a172_optimized_masks"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "a172_phenotype_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RANDOM_SEED = 42

# These must match the feature set used in 07_a172_phenotype_exploration.py.
FEATURE_COLUMNS = [
    "log_area",
    "circularity",
    "solidity",
    "log_aspect_ratio",
    "log_neighbours_within_50px",
    "log_nearest_neighbour_distance",
]

# K is deliberately fixed to the value selected by script 07.  The purpose here is to test the robustness of that selected clustering, not re-select K.
KMEANS_N_INIT = 20
BOOTSTRAP_ITERATIONS = 100

# Image-level confidence intervals describe variation among sampled fields at each timepoint.  They are not biological-replicate confidence intervals when
# only one well/location is present.
TIMEPOINT_BOOTSTRAP_ITERATIONS = 2_000
CONFIDENCE_LEVEL = 0.95
EXEMPLARS_PER_CLUSTER = 6
EXEMPLAR_PADDING_PIXELS = 30

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def require_columns(dataframe, columns, source_name):
    """Stop early if a required output from the earlier pipeline is absent."""

    missing_columns = [column for column in columns if column not in dataframe.columns]

    if missing_columns:
        raise ValueError(
            f"{source_name} is missing required columns: {missing_columns}. "
            "Run 07_a172_phenotype_exploration.py again before phenotype "
            "validation."
        )

def validate_cluster_labels(cells):
    """Return sorted integer cluster labels and check they are usable."""

    if cells["phenotype_cluster"].isna().any():
        raise ValueError("The phenotype-analysis table contains missing cluster labels.")

    cluster_numbers = np.sort(cells["phenotype_cluster"].astype(int).unique())

    if len(cluster_numbers) < 2:
        raise ValueError("At least two phenotype clusters are required for validation.")

    expected_numbers = np.arange(1, len(cluster_numbers) + 1)
    if not np.array_equal(cluster_numbers, expected_numbers):
        raise ValueError(
            "Phenotype clusters must be numbered consecutively from 1. " f"Found: {cluster_numbers.tolist()}"
        )

    return cluster_numbers

def make_scaled_feature_matrix(cells, feature_columns):
    """Scale the phenotype features exactly once for validation analyses."""

    feature_values = cells[feature_columns].to_numpy(dtype=float)

    if not np.isfinite(feature_values).all():
        raise ValueError(
            "The QC-passing phenotype table contains non-finite feature " "values. Re-run script 07 before validation."
        )

    scaler = RobustScaler()
    return scaler.fit_transform(feature_values)

def cluster_centres_by_label(feature_matrix, labels, cluster_numbers):
    """Calculate one mean feature-space centre for each reference cluster."""

    return np.vstack([feature_matrix[labels == cluster_number].mean(axis=0) for cluster_number in cluster_numbers])

def align_bootstrap_labels(raw_labels, bootstrap_centres, reference_centres, cluster_numbers):
    """Align arbitrary K-means labels to the reference cluster numbering."""

    centre_distances = np.linalg.norm(bootstrap_centres[:, np.newaxis, :] - reference_centres[np.newaxis, :, :], axis=2)

    bootstrap_indices, reference_indices = linear_sum_assignment(centre_distances)

    label_mapping = {
        int(bootstrap_index): int(cluster_numbers[reference_index])
        for bootstrap_index, reference_index in zip(bootstrap_indices, reference_indices)
    }

    aligned_labels = np.array([label_mapping[int(label)] for label in raw_labels], dtype=int)
    matched_distances = centre_distances[bootstrap_indices, reference_indices]

    return aligned_labels, matched_distances

def assess_cluster_stability(feature_matrix, reference_labels, cluster_numbers):
    """Refit K-means to bootstrap samples and compare all-cell assignments."""

    generator = np.random.default_rng(RANDOM_SEED)
    reference_centres = cluster_centres_by_label(feature_matrix, reference_labels, cluster_numbers)
    records = []
    n_cells = len(feature_matrix)

    for iteration in range(1, BOOTSTRAP_ITERATIONS + 1):
        bootstrap_indices = generator.integers(low=0, high=n_cells, size=n_cells)
        model = KMeans(n_clusters=len(cluster_numbers), n_init=KMEANS_N_INIT, random_state=RANDOM_SEED + iteration)

        model.fit(feature_matrix[bootstrap_indices])
        raw_labels = model.predict(feature_matrix)

        aligned_labels, matched_distances = align_bootstrap_labels(
            raw_labels, model.cluster_centers_, reference_centres, cluster_numbers
        )

        record = {
            "bootstrap_iteration": iteration,
            "adjusted_rand_index": adjusted_rand_score(reference_labels, raw_labels),
            "mean_centroid_shift": float(matched_distances.mean()),
        }

        for cluster_number in cluster_numbers:
            in_reference_cluster = reference_labels == cluster_number
            record[f"cluster_{cluster_number}_agreement"] = float(
                (aligned_labels[in_reference_cluster] == cluster_number).mean()
            )

        records.append(record)

    stability = pd.DataFrame(records)
    summary_records = [
        {
            "metric": "adjusted_rand_index",
            "mean": stability["adjusted_rand_index"].mean(),
            "median": stability["adjusted_rand_index"].median(),
            "ci_lower": stability["adjusted_rand_index"].quantile(0.025),
            "ci_upper": stability["adjusted_rand_index"].quantile(0.975),
        },
        {
            "metric": "mean_centroid_shift",
            "mean": stability["mean_centroid_shift"].mean(),
            "median": stability["mean_centroid_shift"].median(),
            "ci_lower": stability["mean_centroid_shift"].quantile(0.025),
            "ci_upper": stability["mean_centroid_shift"].quantile(0.975),
        },
    ]

    for cluster_number in cluster_numbers:
        column = f"cluster_{cluster_number}_agreement"
        summary_records.append(
            {
                "metric": f"cluster_{cluster_number}_agreement",
                "mean": stability[column].mean(),
                "median": stability[column].median(),
                "ci_lower": stability[column].quantile(0.025),
                "ci_upper": stability[column].quantile(0.975),
            }
        )

    return stability, pd.DataFrame(summary_records)

def make_image_level_composition(cells, cluster_numbers):
    """Calculate each cluster fraction separately for every image/field."""

    metadata_candidates = ["well", "location", "crop", "elapsed_time", "elapsed_hours"]
    metadata_columns = [column for column in metadata_candidates if column in cells.columns]
    image_metadata = cells.groupby("image")[metadata_columns].first().reset_index()

    image_metadata["_merge_key"] = 1
    cluster_table = pd.DataFrame({"phenotype_cluster": cluster_numbers, "_merge_key": 1})
    image_cluster_grid = image_metadata.merge(cluster_table, on="_merge_key").drop(columns="_merge_key")
    cluster_counts = cells.groupby(["image", "phenotype_cluster"]).size().rename("cells").reset_index()
    total_cells = cells.groupby("image").size().rename("qc_passing_cells_in_image").reset_index()
    composition = image_cluster_grid.merge(cluster_counts, on=["image", "phenotype_cluster"], how="left").merge(
        total_cells, on="image", how="left"
    )

    composition["cells"] = composition["cells"].fillna(0).astype(int)
    composition["cluster_fraction"] = composition["cells"] / composition["qc_passing_cells_in_image"]

    sort_columns = [
        column for column in ["elapsed_hours", "well", "location", "crop", "image"] if column in composition.columns
    ]

    return composition.sort_values(sort_columns + ["phenotype_cluster"]).reset_index(drop=True)

def bootstrap_mean_interval(values, generator):
    """Return a non-parametric interval for the mean of image-level values."""

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    sample_indices = generator.integers(low=0, high=len(values), size=(TIMEPOINT_BOOTSTRAP_ITERATIONS, len(values)))
    bootstrap_means = values[sample_indices].mean(axis=1)
    alpha = (1 - CONFIDENCE_LEVEL) / 2

    return (
        float(values.mean()),
        float(np.quantile(bootstrap_means, alpha)),
        float(np.quantile(bootstrap_means, 1 - alpha)),
    )

def summarise_time_composition(image_composition, cluster_numbers):
    """Summarise field-level cluster fractions for every sampled timepoint."""

    if "elapsed_hours" not in image_composition.columns:
        raise ValueError(
            "The phenotype table has no elapsed_hours metadata. "
            "Run script 07 again with LIVECell-formatted filenames."
        )

    if image_composition["elapsed_hours"].isna().any():
        raise ValueError("Some image filenames could not be converted to elapsed_hours.")

    generator = np.random.default_rng(RANDOM_SEED + 1)
    records = []

    for elapsed_hours in sorted(image_composition["elapsed_hours"].unique()):
        timepoint_data = image_composition.loc[image_composition["elapsed_hours"] == elapsed_hours]
        n_images = int(timepoint_data["image"].nunique())

        for cluster_number in cluster_numbers:
            values = timepoint_data.loc[
                timepoint_data["phenotype_cluster"] == cluster_number, "cluster_fraction"
            ].to_numpy(dtype=float)

            mean_fraction, ci_lower, ci_upper = bootstrap_mean_interval(values, generator)

            records.append(
                {
                    "elapsed_hours": elapsed_hours,
                    "phenotype_cluster": cluster_number,
                    "n_images": n_images,
                    "mean_cluster_fraction": mean_fraction,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                }
            )

    return pd.DataFrame(records)

def compare_earliest_and_latest_timepoints(image_composition, cluster_numbers):
    """Quantify descriptive early-to-late change using image-level bootstrap."""

    generator = np.random.default_rng(RANDOM_SEED + 2)
    timepoints = np.sort(image_composition["elapsed_hours"].unique())

    if len(timepoints) < 2:
        raise ValueError("At least two timepoints are needed to compare change over time.")

    earliest_time = timepoints[0]
    latest_time = timepoints[-1]
    alpha = (1 - CONFIDENCE_LEVEL) / 2
    records = []

    for cluster_number in cluster_numbers:
        early_values = image_composition.loc[
            (image_composition["elapsed_hours"] == earliest_time)
            & (image_composition["phenotype_cluster"] == cluster_number),
            "cluster_fraction",
        ].to_numpy(dtype=float)

        late_values = image_composition.loc[
            (image_composition["elapsed_hours"] == latest_time)
            & (image_composition["phenotype_cluster"] == cluster_number),
            "cluster_fraction",
        ].to_numpy(dtype=float)

        early_indices = generator.integers(
            0, len(early_values), size=(TIMEPOINT_BOOTSTRAP_ITERATIONS, len(early_values))
        )
        late_indices = generator.integers(0, len(late_values), size=(TIMEPOINT_BOOTSTRAP_ITERATIONS, len(late_values)))
        bootstrap_change = late_values[late_indices].mean(axis=1) - early_values[early_indices].mean(axis=1)

        records.append(
            {
                "phenotype_cluster": cluster_number,
                "earliest_time_hours": earliest_time,
                "latest_time_hours": latest_time,
                "mean_fraction_earliest": early_values.mean(),
                "mean_fraction_latest": late_values.mean(),
                "change_latest_minus_earliest": (late_values.mean() - early_values.mean()),
                "bootstrap_ci_lower": np.quantile(bootstrap_change, alpha),
                "bootstrap_ci_upper": np.quantile(bootstrap_change, 1 - alpha),
                "n_images_earliest": len(early_values),
                "n_images_latest": len(late_values),
            }
        )

    return pd.DataFrame(records)

def select_representative_cells(cells, feature_matrix, cluster_numbers):
    """Select prototype-like cells while preferring different source images."""

    working = cells.copy()
    working["_feature_row"] = np.arange(len(working))
    selected_records = []

    for cluster_number in cluster_numbers:
        cluster_cells = working.loc[working["phenotype_cluster"].astype(int) == cluster_number].copy()
        cluster_rows = cluster_cells["_feature_row"].to_numpy(dtype=int)
        cluster_centre = np.median(feature_matrix[cluster_rows], axis=0)

        cluster_cells["prototype_distance"] = np.linalg.norm(feature_matrix[cluster_rows] - cluster_centre, axis=1)

        cluster_cells["image_file_available"] = cluster_cells["image"].map(
            lambda filename: (IMAGE_DIR / filename).exists()
        )
        cluster_cells["mask_file_available"] = cluster_cells["image"].map(
            lambda filename: (MASK_DIR / f"{Path(filename).stem}_optimized_masks.tif").exists()
        )

        candidates = cluster_cells.loc[
            cluster_cells["image_file_available"] & cluster_cells["mask_file_available"]
        ].sort_values("prototype_distance")

        selected_indices = []
        represented_images = set()

        # First pass: spread visual examples across source images where possible.
        for index, row in candidates.iterrows():
            if row["image"] in represented_images:
                continue

            selected_indices.append(index)
            represented_images.add(row["image"])

            if len(selected_indices) == EXEMPLARS_PER_CLUSTER:
                break

        # Second pass: allow extra examples from the same image if needed.
        if len(selected_indices) < EXEMPLARS_PER_CLUSTER:
            for index in candidates.index:
                if index in selected_indices:
                    continue

                selected_indices.append(index)

                if len(selected_indices) == EXEMPLARS_PER_CLUSTER:
                    break

        chosen = candidates.loc[selected_indices].copy()
        chosen["selection_rank"] = np.arange(1, len(chosen) + 1)
        selected_records.append(chosen)

    if len(selected_records) == 0:
        return pd.DataFrame()

    selected = pd.concat(selected_records, ignore_index=True)

    return selected.drop(columns="_feature_row")

def crop_bounds(row, image_shape):
    """Return a padded image crop around a measured cell bounding box."""

    min_row = max(0, int(row["bbox_min_row"]) - EXEMPLAR_PADDING_PIXELS)
    min_col = max(0, int(row["bbox_min_col"]) - EXEMPLAR_PADDING_PIXELS)
    max_row = min(image_shape[0], int(row["bbox_max_row"]) + EXEMPLAR_PADDING_PIXELS)
    max_col = min(image_shape[1], int(row["bbox_max_col"]) + EXEMPLAR_PADDING_PIXELS)

    return min_row, max_row, min_col, max_col

def create_exemplar_figure(selected_cells, cluster_numbers):
    """Plot representative cell crops with their predicted-mask outlines."""

    n_clusters = len(cluster_numbers)
    figure, axes = plt.subplots(
        n_clusters, EXEMPLARS_PER_CLUSTER, figsize=(2.6 * EXEMPLARS_PER_CLUSTER, 2.8 * n_clusters), squeeze=False
    )

    image_cache = {}
    mask_cache = {}
    plot_log = []

    for row_index, cluster_number in enumerate(cluster_numbers):
        cluster_examples = selected_cells.loc[
            selected_cells["phenotype_cluster"].astype(int) == cluster_number
        ].sort_values("selection_rank")

        for column_index in range(EXEMPLARS_PER_CLUSTER):
            axis = axes[row_index, column_index]
            axis.axis("off")

            if column_index >= len(cluster_examples):
                axis.set_title("No available exemplar", fontsize=8)
                continue

            row = cluster_examples.iloc[column_index]
            filename = row["image"]
            image_path = IMAGE_DIR / filename
            mask_path = MASK_DIR / f"{Path(filename).stem}_optimized_masks.tif"

            try:
                if filename not in image_cache:
                    image_cache[filename] = tifffile.imread(image_path)
                    mask_cache[filename] = tifffile.imread(mask_path)

                image = image_cache[filename]
                masks = mask_cache[filename]

                if image.shape != masks.shape:
                    raise ValueError("Image and mask shapes do not match " f"({image.shape} versus {masks.shape}).")

                min_row, max_row, min_col, max_col = crop_bounds(row, image.shape)

                image_crop = image[min_row:max_row, min_col:max_col]
                mask_crop = masks[min_row:max_row, min_col:max_col]
                target_mask = mask_crop == int(row["cell_id"])

                if not target_mask.any():
                    raise ValueError("The selected cell label was not found in its mask.")

                axis.imshow(image_crop, cmap="gray", interpolation="nearest")
                axis.imshow(
                    np.ma.masked_where(~target_mask, target_mask), cmap="spring", alpha=0.35, interpolation="nearest"
                )
                axis.contour(target_mask, levels=[0.5], colors="yellow", linewidths=1.2)

                axis.set_title(
                    f"C{cluster_number} | {row['elapsed_hours']:g} h\n" f"area {row['area']:.0f} pxÃ‚Â²", fontsize=8
                )

                plot_log.append(
                    {
                        "image": filename,
                        "cell_id": int(row["cell_id"]),
                        "phenotype_cluster": cluster_number,
                        "status": "plotted",
                        "message": "",
                    }
                )

            except Exception as error:
                axis.set_title("Exemplar unavailable", fontsize=8)
                plot_log.append(
                    {
                        "image": filename,
                        "cell_id": int(row["cell_id"]),
                        "phenotype_cluster": cluster_number,
                        "status": "failed",
                        "message": str(error),
                    }
                )

    figure.suptitle(
        "Representative exploratory phenotype cells\n"
        "Yellow outline and colour overlay: Cellpose mask for the selected cell",
        y=1.02,
    )
    plt.tight_layout()

    output_path = OUTPUT_DIR / "a172_cluster_exemplar_montage.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")

    return pd.DataFrame(plot_log), output_path

# ---------------------------------------------------------------------
# Load phenotype-analysis data
# --------------------------------------------------------------------

cells = pd.read_csv(INPUT_FILE)
required_columns = [
    "image",
    "cell_id",
    "area",
    "bbox_min_row",
    "bbox_min_col",
    "bbox_max_row",
    "bbox_max_col",
    "elapsed_hours",
    "phenotype_cluster",
    *FEATURE_COLUMNS,
]

require_columns(cells, required_columns, "Phenotype-analysis cell table")

cluster_numbers = validate_cluster_labels(cells)
cells["phenotype_cluster"] = cells["phenotype_cluster"].astype(int)

feature_matrix = make_scaled_feature_matrix(cells, FEATURE_COLUMNS)
reference_labels = cells["phenotype_cluster"].to_numpy(dtype=int)

print(f"QC-passing phenotype cells: {len(cells)}")
print(f"Exploratory clusters:        {len(cluster_numbers)}")
print(f"Images/fields represented:   {cells['image'].nunique()}")

# ---------------------------------------------------------------------
# Validate K-means stability by bootstrap resampling
# ---------------------------------------------------------------------

stability_df, stability_summary_df = assess_cluster_stability(feature_matrix, reference_labels, cluster_numbers)

# ---------------------------------------------------------------------
# Calculate image-level time-course summaries
# ---------------------------------------------------------------------

image_composition_df = make_image_level_composition(cells, cluster_numbers)
time_composition_df = summarise_time_composition(image_composition_df, cluster_numbers)
time_change_df = compare_earliest_and_latest_timepoints(image_composition_df, cluster_numbers)

# ---------------------------------------------------------------------
# Select and plot representative cell/mask examples
# ---------------------------------------------------------------------

exemplar_cells_df = select_representative_cells(cells, feature_matrix, cluster_numbers)

if len(exemplar_cells_df) == 0:
    raise RuntimeError("No exemplar cells could be selected. Check IMAGE_DIR and MASK_DIR.")

exemplar_log_df, exemplar_figure_path = create_exemplar_figure(exemplar_cells_df, cluster_numbers)

# ---------------------------------------------------------------------
# Save validation tables and manifest
# ---------------------------------------------------------------------

stability_output_path = OUTPUT_DIR / "a172_cluster_bootstrap_stability.csv"
stability_summary_output_path = OUTPUT_DIR / "a172_cluster_stability_summary.csv"
image_composition_output_path = OUTPUT_DIR / "a172_cluster_composition_by_image.csv"
time_composition_output_path = OUTPUT_DIR / "a172_cluster_composition_by_time_with_ci.csv"
time_change_output_path = OUTPUT_DIR / "a172_cluster_change_earliest_to_latest.csv"
exemplar_cells_output_path = OUTPUT_DIR / "a172_cluster_exemplar_cells.csv"
exemplar_log_output_path = OUTPUT_DIR / "a172_exemplar_plot_log.csv"
manifest_output_path = OUTPUT_DIR / "phenotype_validation_manifest.json"

stability_df.to_csv(stability_output_path, index=False)
stability_summary_df.to_csv(stability_summary_output_path, index=False)
image_composition_df.to_csv(image_composition_output_path, index=False)
time_composition_df.to_csv(time_composition_output_path, index=False)
time_change_df.to_csv(time_change_output_path, index=False)
exemplar_cells_df.to_csv(exemplar_cells_output_path, index=False)
exemplar_log_df.to_csv(exemplar_log_output_path, index=False)

manifest = {
    "analysis": "A172 exploratory phenotype validation",
    "input_file": str(INPUT_FILE),
    "input_cells": int(len(cells)),
    "input_images_fields": int(cells["image"].nunique()),
    "cluster_numbers": [int(number) for number in cluster_numbers],
    "feature_columns": FEATURE_COLUMNS,
    "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
    "timepoint_bootstrap_iterations": TIMEPOINT_BOOTSTRAP_ITERATIONS,
    "confidence_level": CONFIDENCE_LEVEL,
    "exemplars_per_cluster_requested": EXEMPLARS_PER_CLUSTER,
    "exemplars_plotted": int((exemplar_log_df["status"] == "plotted").sum()),
    "exemplars_failed": int((exemplar_log_df["status"] != "plotted").sum()),
    "interpretation": (
        "Bootstrap stability tests whether the clustering is internally "
        "robust to resampling cells. Time-course intervals are based on "
        "images/fields and are not biological-replicate inference when "
        "only one well/location is represented."
    ),
}

with manifest_output_path.open("w", encoding="utf-8") as output_file:
    json.dump(manifest, output_file, indent=4)

# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

ari_summary = stability_summary_df.loc[stability_summary_df["metric"] == "adjusted_rand_index"].iloc[0]

print("\n")
print("=" * 72)
print("A172 EXPLORATORY PHENOTYPE VALIDATION")
print("=" * 72)
print(f"QC-passing cells:           {len(cells)}")
print(f"Images/fields:              {cells['image'].nunique()}")
print(f"Bootstrap refits:           {BOOTSTRAP_ITERATIONS}")
print(
    "Bootstrap ARI (median, 95% interval): "
    f"{ari_summary['median']:.3f} "
    f"({ari_summary['ci_lower']:.3f} to {ari_summary['ci_upper']:.3f})"
)

print("\nCluster stability summary:")
print(stability_summary_df.round(3).to_string(index=False))

print("\nImage-level earliest-to-latest change:")
print(time_change_df.round(3).to_string(index=False))

print(f"\nSaved stability results to:\n{stability_output_path}")
print(f"\nSaved image-level composition to:\n{image_composition_output_path}")
print(f"\nSaved exemplar montage to:\n{exemplar_figure_path}")

# ---------------------------------------------------------------------
# Plot: stability distribution and image-level time-course uncertainty
# ---------------------------------------------------------------------

figure, axes = plt.subplots(1, 2, figsize=(15, 5.5))

axes[0].hist(stability_df["adjusted_rand_index"], bins=20, color="tab:blue", alpha=0.8, edgecolor="white")
axes[0].axvline(
    stability_df["adjusted_rand_index"].median(),
    color="tab:orange",
    linestyle="--",
    label=("Median = " f"{stability_df['adjusted_rand_index'].median():.3f}"),
)
axes[0].set_title("Cluster stability across bootstrap refits")
axes[0].set_xlabel("Adjusted Rand index versus reference clustering")
axes[0].set_ylabel("Bootstrap refits")
axes[0].legend()

for cluster_number in cluster_numbers:
    cluster_time_data = time_composition_df.loc[time_composition_df["phenotype_cluster"] == cluster_number].sort_values(
        "elapsed_hours"
    )

    axes[1].plot(
        cluster_time_data["elapsed_hours"],
        cluster_time_data["mean_cluster_fraction"],
        marker="o",
        label=f"Morphology cluster {cluster_number}",
    )
    axes[1].fill_between(
        cluster_time_data["elapsed_hours"], cluster_time_data["ci_lower"], cluster_time_data["ci_upper"], alpha=0.2
    )

axes[1].set_title("Image-level cluster composition over time")
axes[1].set_xlabel("Elapsed time (hours)")
axes[1].set_ylabel("Mean fraction of QC-passing cells per image")
axes[1].set_ylim(0, 1)
axes[1].legend()

figure.suptitle("A172 phenotype validation: internal stability and field-level variation")
plt.tight_layout()

summary_figure_path = OUTPUT_DIR / "a172_cluster_stability_and_timecourse.png"

plt.savefig(summary_figure_path, dpi=200, bbox_inches="tight")
