"""Explore A172 single-cell morphology and crowding phenotypes.

This script reads the single-cell table from 06_large_scale_a172_analysis.py.
It applies transparent quality-control flags, creates a non-redundant feature
matrix, performs PCA, and uses K-means clustering to identify exploratory
morphology/crowding groups.

The clusters are not asserted to be biological cell states.  They are a
reproducible description of patterns in the measured features that can guide
later validation and multi-cell-line phenotype classification.
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path().resolve().parent
INPUT_FILE = PROJECT_ROOT / "outputs" / "a172_large_scale_analysis" / "a172_single_cell_features.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "a172_phenotype_exploration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RANDOM_SEED = 42

# Cells below or above these global area quantiles are marked as extreme and excluded from clustering.  They remain in the complete QC output table.
LOWER_AREA_QUANTILE = 0.01
UPPER_AREA_QUANTILE = 0.99

# K is selected from this range using the highest silhouette score.
MIN_CLUSTERS = 2
MAX_CLUSTERS = 6
KMEANS_N_INIT = 20
SILHOUETTE_SAMPLE_SIZE = 10_000
PLOT_MAX_CELLS = 5_000

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def parse_livecell_filename(filename):
    """Extract metadata from the documented LIVECell filename convention."""

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

def add_filename_metadata(cells):
    """Recreate metadata from image names, including for pre-patch CSV files."""

    metadata = pd.DataFrame([parse_livecell_filename(filename) for filename in cells["image"]], index=cells.index)

    for column in metadata.columns:
        cells[column] = metadata[column]

    return cells

def require_columns(dataframe, columns):
    """Stop early with a useful message when a previous output is incomplete."""

    missing_columns = [column for column in columns if column not in dataframe.columns]

    if missing_columns:
        raise ValueError(
            "The single-cell table is missing required columns: "
            f"{missing_columns}. Run 06_large_scale_a172_analysis.py "
            "again before phenotype exploration."
        )

def add_derived_features(cells):
    """Create transformed features used for the exploratory phenotype model."""

    cells = cells.copy()

    cells["log_area"] = np.where(cells["area"] > 0, np.log(cells["area"]), np.nan)

    cells["log_aspect_ratio"] = np.where(cells["aspect_ratio"] > 0, np.log(cells["aspect_ratio"]), np.nan)

    cells["log_neighbours_within_50px"] = np.log1p(cells["neighbours_within_50px"])

    cells["log_nearest_neighbour_distance"] = np.where(
        cells["nearest_neighbour_distance"] > 0, np.log(cells["nearest_neighbour_distance"]), np.nan
    )

    return cells

def add_qc_flags(cells, feature_columns):
    """Flag potential measurement issues without deleting any source cells."""

    cells = cells.copy()
    finite_area = cells.loc[np.isfinite(cells["area"]), "area"]
    lower_area_limit = finite_area.quantile(LOWER_AREA_QUANTILE)
    upper_area_limit = finite_area.quantile(UPPER_AREA_QUANTILE)

    # CSV files can represent booleans as either true booleans or strings. ``astype(bool)`` is unsafe for strings because bool("False") is True.
    border_values = cells["touches_image_border"]
    if pd.api.types.is_bool_dtype(border_values):
        cells["qc_touches_image_border"] = border_values.fillna(False)
    else:
        cells["qc_touches_image_border"] = border_values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})

    cells["qc_extreme_area"] = (cells["area"] < lower_area_limit) | (cells["area"] > upper_area_limit)

    feature_values = cells[feature_columns].to_numpy(dtype=float)

    cells["qc_missing_feature"] = ~np.isfinite(feature_values).all(axis=1)

    cells["qc_include_phenotype"] = ~(
        cells["qc_touches_image_border"] | cells["qc_extreme_area"] | cells["qc_missing_feature"]
    )

    # Keep all applicable reasons.  Build the strings explicitly rather than stacking a table: older pandas versions may convert missing entries to
    # floating-point NaN, which cannot be joined with text labels.
    exclusion_reason = pd.Series("", index=cells.index, dtype="object")

    for condition, reason in [
        (cells["qc_touches_image_border"], "touches_image_border"),
        (cells["qc_extreme_area"], "extreme_area"),
        (cells["qc_missing_feature"], "missing_feature"),
    ]:
        appended_reason = np.where(exclusion_reason.eq(""), reason, exclusion_reason + ";" + reason)
        exclusion_reason = exclusion_reason.where(~condition, appended_reason)

    cells["qc_exclusion_reason"] = exclusion_reason.replace("", "included")

    return cells, lower_area_limit, upper_area_limit

def select_cluster_count(feature_matrix):
    """Fit candidate K-means models and select K by silhouette score."""

    maximum_clusters = min(MAX_CLUSTERS, len(feature_matrix) - 1)

    if maximum_clusters < MIN_CLUSTERS:
        raise ValueError("Too few QC-passing cells for clustering. " f"Found {len(feature_matrix)} cells.")

    candidate_results = []

    for n_clusters in range(MIN_CLUSTERS, maximum_clusters + 1):
        model = KMeans(n_clusters=n_clusters, n_init=KMEANS_N_INIT, random_state=RANDOM_SEED)
        labels = model.fit_predict(feature_matrix)
        silhouette = silhouette_score(
            feature_matrix,
            labels,
            sample_size=min(SILHOUETTE_SAMPLE_SIZE, len(feature_matrix)),
            random_state=RANDOM_SEED,
        )

        candidate_results.append({"n_clusters": n_clusters, "silhouette_score": silhouette, "inertia": model.inertia_})

    candidate_df = pd.DataFrame(candidate_results)
    best_n_clusters = int(candidate_df.loc[candidate_df["silhouette_score"].idxmax(), "n_clusters"])

    return candidate_df, best_n_clusters

def order_clusters_by_area(cells, raw_labels):
    """Give clusters stable labels from smallest to largest median cell area."""

    label_table = pd.DataFrame({"raw_cluster": raw_labels, "area": cells["area"].to_numpy()})
    ordered_raw_clusters = label_table.groupby("raw_cluster")["area"].median().sort_values().index.tolist()
    label_mapping = {
        raw_cluster: cluster_number for cluster_number, raw_cluster in enumerate(ordered_raw_clusters, start=1)
    }

    ordered_labels = np.array([label_mapping[label] for label in raw_labels], dtype=int)

    return ordered_labels, label_mapping

def make_cluster_summary(cells, raw_feature_columns):
    """Describe each exploratory cluster in original measurement units."""

    summary = (
        cells.groupby("phenotype_cluster")
        .agg(
            cells=("cell_id", "size"),
            median_area=("area", "median"),
            median_circularity=("circularity", "median"),
            median_solidity=("solidity", "median"),
            median_aspect_ratio=("aspect_ratio", "median"),
            median_neighbours=("neighbours_within_50px", "median"),
            median_nearest_neighbour_distance=("nearest_neighbour_distance", "median"),
            border_touching_fraction=("qc_touches_image_border", "mean"),
        )
        .reset_index()
    )

    summary["cell_fraction"] = summary["cells"] / summary["cells"].sum()

    profile = cells.groupby("phenotype_cluster")[raw_feature_columns].median()

    return summary, profile

def make_acquisition_composition(cells):
    """Calculate cluster composition per well/location/time acquisition."""

    grouping_columns = ["well", "location", "elapsed_hours"]
    composition = cells.groupby(grouping_columns + ["phenotype_cluster"]).size().rename("cells").reset_index()

    composition["total_cells_in_acquisition"] = composition.groupby(grouping_columns)["cells"].transform("sum")

    composition["cluster_fraction"] = composition["cells"] / composition["total_cells_in_acquisition"]

    return composition

# ---------------------------------------------------------------------
# Load single-cell feature data
# ---------------------------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        "Single-cell features were not found. Run "
        "06_large_scale_a172_analysis.py first.\n"
        f"Expected file: {INPUT_FILE}"
    )

cells = pd.read_csv(INPUT_FILE)
required_columns = [
    "image",
    "cell_id",
    "area",
    "circularity",
    "solidity",
    "aspect_ratio",
    "neighbours_within_50px",
    "nearest_neighbour_distance",
    "touches_image_border",
]

require_columns(cells, required_columns)

cells = add_filename_metadata(cells)
cells = add_derived_features(cells)
feature_columns = [
    "log_area",
    "circularity",
    "solidity",
    "log_aspect_ratio",
    "log_neighbours_within_50px",
    "log_nearest_neighbour_distance",
]

raw_feature_columns = [
    "area",
    "circularity",
    "solidity",
    "aspect_ratio",
    "neighbours_within_50px",
    "nearest_neighbour_distance",
]

cells, lower_area_limit, upper_area_limit = add_qc_flags(cells, feature_columns)

# ---------------------------------------------------------------------
# Prepare phenotype feature matrix
# ---------------------------------------------------------------------

analysis_cells = cells.loc[cells["qc_include_phenotype"]].copy()

if len(analysis_cells) == 0:
    raise RuntimeError("No cells passed phenotype-analysis quality control.")

scaler = RobustScaler()
scaled_features = scaler.fit_transform(analysis_cells[feature_columns])
pca = PCA()
pca_scores = pca.fit_transform(scaled_features)

analysis_cells["PC1"] = pca_scores[:, 0]
analysis_cells["PC2"] = pca_scores[:, 1]

explained_variance_df = pd.DataFrame(
    {
        "principal_component": [f"PC{index}" for index in range(1, len(pca.explained_variance_ratio_) + 1)],
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
    }
)

pca_loadings_df = pd.DataFrame(
    pca.components_.T, index=feature_columns, columns=explained_variance_df["principal_component"]
).reset_index(names="feature")

# ---------------------------------------------------------------------
# Select K-means cluster count and fit final model
# ---------------------------------------------------------------------

cluster_selection_df, best_n_clusters = select_cluster_count(scaled_features)

final_kmeans = KMeans(n_clusters=best_n_clusters, n_init=KMEANS_N_INIT, random_state=RANDOM_SEED)
raw_cluster_labels = final_kmeans.fit_predict(scaled_features)

phenotype_labels, cluster_label_mapping = order_clusters_by_area(analysis_cells, raw_cluster_labels)

analysis_cells["phenotype_cluster"] = phenotype_labels
analysis_cells["phenotype_cluster_label"] = [f"Morphology cluster {label}" for label in phenotype_labels]

# Reindexing copies results back to the full cell table while retaining a compatible dtype for each output.  In particular, the text label column
# must not first be created as a float column filled with ``np.nan``.
cells["PC1"] = analysis_cells["PC1"].reindex(cells.index)
cells["PC2"] = analysis_cells["PC2"].reindex(cells.index)
cells["phenotype_cluster"] = analysis_cells["phenotype_cluster"].astype("Int64").reindex(cells.index)
cells["phenotype_cluster_label"] = analysis_cells["phenotype_cluster_label"].astype("object").reindex(cells.index)

# ---------------------------------------------------------------------
# Summarise exploratory phenotype groups
# ---------------------------------------------------------------------

cluster_summary_df, cluster_profile_df = make_cluster_summary(analysis_cells, raw_feature_columns)

global_feature_medians = analysis_cells[raw_feature_columns].median()
global_feature_iqrs = analysis_cells[raw_feature_columns].quantile(0.75) - analysis_cells[raw_feature_columns].quantile(
    0.25
)

cluster_profile_relative_df = cluster_profile_df.subtract(global_feature_medians, axis="columns").divide(
    global_feature_iqrs.replace(0, np.nan), axis="columns"
)

cluster_composition_df = make_acquisition_composition(analysis_cells)
observed_time_composition_df = (
    cluster_composition_df.groupby(["elapsed_hours", "phenotype_cluster"])["cells"].sum().rename("cells").reset_index()
)

# Explicit zeros make the time-course plot and CSV unambiguous when a morphology cluster is absent at a sampled timepoint.
observed_times = sorted(analysis_cells["elapsed_hours"].dropna().unique())
cluster_numbers = range(1, best_n_clusters + 1)
time_index = pd.MultiIndex.from_product([observed_times, cluster_numbers], names=["elapsed_hours", "phenotype_cluster"])
time_composition_df = (
    observed_time_composition_df.set_index(["elapsed_hours", "phenotype_cluster"])
    .reindex(time_index, fill_value=0)
    .reset_index()
)

time_composition_df["cells_at_timepoint"] = time_composition_df.groupby("elapsed_hours")["cells"].transform("sum")

time_composition_df["cluster_fraction"] = time_composition_df["cells"] / time_composition_df["cells_at_timepoint"]

# ---------------------------------------------------------------------
# Save tables and analysis manifest
# ---------------------------------------------------------------------

all_cells_output_path = OUTPUT_DIR / "a172_cells_with_qc_and_clusters.csv"
analysis_cells_output_path = OUTPUT_DIR / "a172_phenotype_analysis_cells.csv"
cluster_selection_output_path = OUTPUT_DIR / "a172_cluster_selection.csv"
cluster_summary_output_path = OUTPUT_DIR / "a172_cluster_summary.csv"
cluster_profile_output_path = OUTPUT_DIR / "a172_cluster_feature_profile.csv"
composition_output_path = OUTPUT_DIR / "a172_cluster_composition_by_acquisition.csv"
time_composition_output_path = OUTPUT_DIR / "a172_cluster_composition_by_time.csv"
variance_output_path = OUTPUT_DIR / "a172_pca_variance.csv"
loadings_output_path = OUTPUT_DIR / "a172_pca_loadings.csv"
manifest_output_path = OUTPUT_DIR / "phenotype_exploration_manifest.json"

cells.to_csv(all_cells_output_path, index=False)
analysis_cells.to_csv(analysis_cells_output_path, index=False)
cluster_selection_df.to_csv(cluster_selection_output_path, index=False)
cluster_summary_df.to_csv(cluster_summary_output_path, index=False)
cluster_profile_relative_df.to_csv(cluster_profile_output_path)
cluster_composition_df.to_csv(composition_output_path, index=False)
time_composition_df.to_csv(time_composition_output_path, index=False)
explained_variance_df.to_csv(variance_output_path, index=False)
pca_loadings_df.to_csv(loadings_output_path, index=False)

manifest = {
    "analysis": "Exploratory A172 morphology and crowding phenotypes",
    "input_file": str(INPUT_FILE),
    "random_seed": RANDOM_SEED,
    "input_cells": int(len(cells)),
    "qc_passing_cells": int(len(analysis_cells)),
    "qc_excluded_cells": int(len(cells) - len(analysis_cells)),
    "area_quantile_limits": {
        "lower_quantile": LOWER_AREA_QUANTILE,
        "upper_quantile": UPPER_AREA_QUANTILE,
        "lower_area_pixels": float(lower_area_limit),
        "upper_area_pixels": float(upper_area_limit),
    },
    "phenotype_features": feature_columns,
    "excluded_features": {
        "intensity": (
            "Raw phase-contrast intensity is not included because it may " "reflect illumination and imaging variation."
        ),
        "equivalent_diameter_area": "Redundant with area.",
        "eccentricity": "Overlaps with aspect ratio and circularity.",
    },
    "clustering_algorithm": "KMeans",
    "candidate_cluster_counts": list(cluster_selection_df["n_clusters"].astype(int)),
    "selected_cluster_count": int(best_n_clusters),
    "cluster_ordering": "ascending median cell area",
    "interpretation": (
        "Clusters are exploratory morphology/crowding patterns, not " "validated biological cell states."
    ),
}

with manifest_output_path.open("w", encoding="utf-8") as output_file:
    json.dump(manifest, output_file, indent=4)

# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

print("\n")
print("=" * 72)
print("A172 EXPLORATORY PHENOTYPE ANALYSIS")
print("=" * 72)

print(f"Input cells:               {len(cells)}")
print(f"QC-passing cells:          {len(analysis_cells)}")
print(f"Excluded from clustering:  {len(cells) - len(analysis_cells)}")
print("Area QC limits:            " f"{lower_area_limit:.1f} to {upper_area_limit:.1f} pixels")
print(f"Selected cluster count:    {best_n_clusters}")
print("PC1 + PC2 variance:        " f"{explained_variance_df.loc[:1, 'explained_variance_ratio'].sum():.1%}")

print("\nCluster-selection results:")
print(cluster_selection_df.round(3).to_string(index=False))

print("\nExploratory cluster summary:")
print(cluster_summary_df.round(3).to_string(index=False))

print(f"\nSaved all cells with QC flags to:\n{all_cells_output_path}")
print(f"\nSaved QC-passing phenotype cells to:\n{analysis_cells_output_path}")
print(f"\nSaved cluster summary to:\n{cluster_summary_output_path}")

# ---------------------------------------------------------------------
# Plot 1: quality control, cluster selection, and PCA
# ---------------------------------------------------------------------

if len(analysis_cells) > PLOT_MAX_CELLS:
    plot_cells = analysis_cells.sample(PLOT_MAX_CELLS, random_state=RANDOM_SEED)
else:
    plot_cells = analysis_cells
cluster_colours = plt.get_cmap("tab10")((plot_cells["phenotype_cluster"].astype(int) - 1) % 10)

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

qc_counts = pd.Series(
    {
        "Input": len(cells),
        "Border": int(cells["qc_touches_image_border"].sum()),
        "Extreme area": int(cells["qc_extreme_area"].sum()),
        "Missing feature": int(cells["qc_missing_feature"].sum()),
        "Included": int(cells["qc_include_phenotype"].sum()),
    }
)

axes[0, 0].bar(
    qc_counts.index, qc_counts.values, color=["tab:gray", "tab:red", "tab:orange", "tab:purple", "tab:green"]
)
axes[0, 0].set_title("Quality-control accounting")
axes[0, 0].set_ylabel("Cells")
axes[0, 0].tick_params(axis="x", rotation=25)

axes[0, 1].plot(cluster_selection_df["n_clusters"], cluster_selection_df["silhouette_score"], marker="o")
axes[0, 1].axvline(best_n_clusters, color="tab:orange", linestyle="--", label=f"Selected K = {best_n_clusters}")
axes[0, 1].set_title("K-means cluster selection")
axes[0, 1].set_xlabel("Number of clusters")
axes[0, 1].set_ylabel("Silhouette score")
axes[0, 1].legend()

axes[1, 0].scatter(plot_cells["PC1"], plot_cells["PC2"], c=cluster_colours, s=12, alpha=0.35)
axes[1, 0].set_title("PCA coloured by exploratory cluster")
axes[1, 0].set_xlabel(f"PC1 ({explained_variance_df.loc[0, 'explained_variance_ratio']:.1%})")
axes[1, 0].set_ylabel(f"PC2 ({explained_variance_df.loc[1, 'explained_variance_ratio']:.1%})")

pc_loadings = pca_loadings_df.set_index("feature")[["PC1", "PC2"]]

pc_loadings.plot.barh(ax=axes[1, 1])
axes[1, 1].set_title("Feature loadings on PC1 and PC2")
axes[1, 1].set_xlabel("Loading")
axes[1, 1].axvline(0, color="black", linewidth=0.8)

fig.suptitle("A172 exploratory phenotype analysis")
plt.tight_layout()

overview_figure_path = OUTPUT_DIR / "a172_qc_cluster_selection_pca.png"

plt.savefig(overview_figure_path, dpi=200, bbox_inches="tight")

# ---------------------------------------------------------------------
# Plot 2: relative cluster feature profiles
# ---------------------------------------------------------------------

plt.figure(figsize=(11, max(4, 0.8 * best_n_clusters + 2)))

image = plt.imshow(cluster_profile_relative_df, cmap="coolwarm", aspect="auto", vmin=-2, vmax=2)

plt.colorbar(image, label="Median difference from all cells (IQR units)")

plt.xticks(
    range(len(cluster_profile_relative_df.columns)), cluster_profile_relative_df.columns, rotation=25, ha="right"
)

plt.yticks(
    range(len(cluster_profile_relative_df.index)),
    [f"Morphology cluster {cluster}" for cluster in cluster_profile_relative_df.index],
)

plt.title("Relative morphology and crowding profiles")
plt.tight_layout()

profile_figure_path = OUTPUT_DIR / "a172_cluster_feature_profiles.png"

plt.savefig(profile_figure_path, dpi=200, bbox_inches="tight")

# ---------------------------------------------------------------------
# Plot 3: exploratory cluster composition over time
# ---------------------------------------------------------------------

plt.figure(figsize=(10, 6))

for cluster in sorted(time_composition_df["phenotype_cluster"].unique()):
    cluster_data = time_composition_df.loc[time_composition_df["phenotype_cluster"] == cluster].sort_values(
        "elapsed_hours"
    )

    plt.plot(
        cluster_data["elapsed_hours"],
        cluster_data["cluster_fraction"],
        marker="o",
        label=f"Morphology cluster {cluster}",
    )

plt.xlabel("Elapsed time (hours)")
plt.ylabel("Fraction of QC-passing cells")
plt.title("Exploratory cluster composition over time")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()

time_figure_path = OUTPUT_DIR / "a172_cluster_composition_over_time.png"

plt.savefig(time_figure_path, dpi=200, bbox_inches="tight")