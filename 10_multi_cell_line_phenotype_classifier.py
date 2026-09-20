"""Classify LIVECell cell line from single-cell morphology and crowding.

This script follows 09_multi_cell_line_feature_extraction.py. It evaluates
whether single-cell morphology and local crowding features distinguish the
five selected LIVECell cell lines.  It uses group-aware cross-validation: all
cells from a held-out field remain out of the corresponding training fold.

The goal is a reproducible phenotype classifier, not a diagnostic assay or a
claim that morphology alone establishes biological identity.
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path().resolve().parent
INPUT_FILE = PROJECT_ROOT / "outputs" / "multi_cell_line_analysis" / "multicell_line_classifier_cells.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "multi_cell_line_classifier"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RANDOM_SEED = 42

# The preferred grouping is a well/location field, so that different time points and crops from the same field cannot leak into train and test data.
# If this produces fewer than two groups for any line, the script falls back to grouping by image and reports this explicitly.
DESIRED_CV_FOLDS = 5

# Each training fold is balanced by downsampling all lines to the smallest available line in that fold, capped for runtime. Held-out test data are
# never downsampled: the reported performance covers every available test cell.
MAX_TRAINING_CELLS_PER_CLASS = 5_000
LOGISTIC_MAX_ITERATIONS = 2_000
RANDOM_FOREST_TREES = 300
RANDOM_FOREST_MIN_SAMPLES_LEAF = 5

# ---------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------

# Raw intensity is intentionally excluded because it can capture illumination and acquisition differences rather than cell phenotype.
RAW_FEATURE_COLUMNS = [
    "area",
    "circularity",
    "solidity",
    "aspect_ratio",
    "neighbours_within_50px",
    "nearest_neighbour_distance",
]

FEATURE_COLUMNS = [
    "log_area",
    "circularity",
    "solidity",
    "log_aspect_ratio",
    "log_neighbours_within_50px",
    "log_nearest_neighbour_distance",
]

TARGET_COLUMN = "cell_line"

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def require_columns(dataframe, columns):
    """Stop early with a useful message if script 09 output is incomplete."""

    missing_columns = [column for column in columns if column not in dataframe.columns]

    if missing_columns:
        raise ValueError(
            "The classifier input table is missing required columns: "
            f"{missing_columns}. Run "
            "09_multi_cell_line_feature_extraction.py again first."
        )

def add_classifier_features(cells):
    """Create the transformed morphology/crowding classifier feature set."""

    cells = cells.copy()

    cells["log_area"] = np.where(cells["area"] > 0, np.log(cells["area"]), np.nan)
    cells["log_aspect_ratio"] = np.where(cells["aspect_ratio"] > 0, np.log(cells["aspect_ratio"]), np.nan)
    cells["log_neighbours_within_50px"] = np.where(cells["neighbours_within_50px"] >= 0, np.log1p(cells["neighbours_within_50px"]), np.nan)
    cells["log_nearest_neighbour_distance"] = np.where(cells["nearest_neighbour_distance"] > 0, np.log(cells["nearest_neighbour_distance"]), np.nan)

    return cells

def make_split_groups(cells):
    """Create strict field-level groups, with an image-level fallback."""

    has_field_metadata = (
        "well" in cells.columns
        and "location" in cells.columns
        and not cells["well"].isna().any()
        and not cells["location"].isna().any())

    if has_field_metadata:
        field_groups = (
            cells[TARGET_COLUMN].astype(str) + "|" + cells["well"].astype(str) + "|" + cells["location"].astype(str))

        group_counts = (
            pd.DataFrame({TARGET_COLUMN: cells[TARGET_COLUMN], "split_group": field_groups})
            .drop_duplicates()
            .groupby(TARGET_COLUMN)
            .size())

        if group_counts.min() >= 2:
            return field_groups, "cell_line + well + location", group_counts

    image_groups = cells[TARGET_COLUMN].astype(str) + "|" + cells["image"].astype(str)
    group_counts = (
        pd.DataFrame({TARGET_COLUMN: cells[TARGET_COLUMN], "split_group": image_groups})
        .drop_duplicates()
        .groupby(TARGET_COLUMN)
        .size())

    if group_counts.min() < 2:
        raise ValueError(
            "At least two independent image/field groups per cell line are "
            "needed for group-aware classifier validation.\n"
            f"Groups per line: {group_counts.to_dict()}")

    return image_groups, "cell_line + image", group_counts

def balanced_training_indices(train_indices, labels, class_labels, generator):
    """Downsample each cell line equally within a training fold."""

    per_class_indices = {}

    for class_label in class_labels:
        class_indices = train_indices[labels[train_indices] == class_label]

        if len(class_indices) == 0:
            raise ValueError(f"Training fold has no cells from {class_label}.")

        per_class_indices[class_label] = class_indices

    cells_per_class = min(min(len(indices) for indices in per_class_indices.values()), MAX_TRAINING_CELLS_PER_CLASS)
    selected = []

    for class_label in class_labels:
        selected.append(generator.choice(per_class_indices[class_label], size=cells_per_class, replace=False))

    return np.concatenate(selected), cells_per_class

def make_models(random_state):
    """Return an interpretable baseline and a non-linear phenotype model."""

    return {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", RobustScaler()),
                ("classifier", LogisticRegression(max_iter=LOGISTIC_MAX_ITERATIONS, random_state=random_state)),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=RANDOM_FOREST_TREES,
            min_samples_leaf=RANDOM_FOREST_MIN_SAMPLES_LEAF,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
    }

def calculate_metrics(true_labels, predicted_labels, class_labels):
    """Calculate overall and class-specific metrics without class imbalance bias."""

    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predicted_labels, labels=class_labels, zero_division=0
    )

    overall = {
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "balanced_accuracy": balanced_accuracy_score(true_labels, predicted_labels),
        "macro_f1": f1_score(true_labels, predicted_labels, labels=class_labels, average="macro", zero_division=0),
    }

    class_metrics = pd.DataFrame(
        {"cell_line": class_labels, "precision": precision, "recall": recall, "f1": f1, "support": support}
    )

    return overall, class_metrics

def create_prediction_table(cells, test_indices, fold_number, model_name, model, class_labels, predictions):
    """Record held-out predictions and class probabilities for auditability."""

    prediction_table = cells.iloc[test_indices][
        ["input_cell_index", "image", "well", "location", "elapsed_hours", "split_group", TARGET_COLUMN]
    ].copy()

    prediction_table = prediction_table.rename(columns={TARGET_COLUMN: "true_cell_line"})
    prediction_table["fold"] = fold_number
    prediction_table["model"] = model_name
    prediction_table["predicted_cell_line"] = predictions
    prediction_table["correct"] = prediction_table["true_cell_line"] == prediction_table["predicted_cell_line"]

    probabilities = model.predict_proba(cells.iloc[test_indices][FEATURE_COLUMNS].to_numpy(dtype=float))

    for class_index, class_label in enumerate(model.classes_):
        prediction_table[f"probability_{class_label}"] = probabilities[:, class_index]

    for class_label in class_labels:
        probability_column = f"probability_{class_label}"
        if probability_column not in prediction_table.columns:
            prediction_table[probability_column] = 0.0

    return prediction_table

def plot_confusion_matrix(axis, matrix, class_labels, title):
    """Plot a row-normalised confusion matrix with readable cell labels."""

    image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1)

    axis.set_xticks(np.arange(len(class_labels)), class_labels, rotation=30, ha="right")
    axis.set_yticks(np.arange(len(class_labels)), class_labels)
    axis.set_xlabel("Predicted cell line")
    axis.set_ylabel("True cell line")
    axis.set_title(title)

    for row_index in range(len(class_labels)):
        for column_index in range(len(class_labels)):
            value = matrix[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.55 else "black",
                fontsize=8,
            )

    return image

# ---------------------------------------------------------------------
# Load and prepare classifier input
# ---------------------------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        "Classifier-ready cells were not found. Run "
        "09_multi_cell_line_feature_extraction.py first.\n"
        f"Expected file: {INPUT_FILE}"
    )

cells = pd.read_csv(INPUT_FILE)
required_columns = ["image", TARGET_COLUMN, "well", "location", "elapsed_hours", *RAW_FEATURE_COLUMNS]

require_columns(cells, required_columns)

cells = add_classifier_features(cells)
cells["input_cell_index"] = np.arange(len(cells))

finite_features = np.isfinite(cells[FEATURE_COLUMNS].to_numpy(dtype=float)).all(axis=1)

if not finite_features.all():
    removed_cells = int((~finite_features).sum())
    print("WARNING: removing " f"{removed_cells} cells with non-finite classifier features.")
    cells = cells.loc[finite_features].copy()
class_labels = np.sort(cells[TARGET_COLUMN].unique())

if len(class_labels) < 3:
    raise ValueError("At least three cell lines are needed for the multi-cell-line " "classifier.")

cells["split_group"], grouping_strategy, group_counts = make_split_groups(cells)

cv_folds = min(DESIRED_CV_FOLDS, int(group_counts.min()))

if cv_folds < 2:
    raise ValueError("Fewer than two valid group-aware folds are available.")

features = cells[FEATURE_COLUMNS].to_numpy(dtype=float)
labels = cells[TARGET_COLUMN].to_numpy()
groups = cells["split_group"].to_numpy()

print(f"Input QC-passing cells:     {len(cells)}")
print(f"Cell lines:                 {', '.join(class_labels)}")
print(f"Split grouping:             {grouping_strategy}")
print(f"Groups per cell line:       {group_counts.to_dict()}")
print(f"Cross-validation folds:     {cv_folds}")

# ---------------------------------------------------------------------
# Group-aware cross-validation and model comparison
# ---------------------------------------------------------------------

splitter = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
fold_metrics = []
class_metrics = []
prediction_tables = []
feature_importance_tables = []
training_records = []
fold_iterator = splitter.split(features, labels, groups)

for fold_number, (train_indices, test_indices) in enumerate(
    tqdm(fold_iterator, total=cv_folds, desc="Group-aware classifier cross-validation"), start=1
):
    train_groups = set(groups[train_indices])
    test_groups = set(groups[test_indices])

    if train_groups.intersection(test_groups):
        raise RuntimeError("A group was found in both train and test data; stopping to " "prevent leakage.")

    generator = np.random.default_rng(RANDOM_SEED + fold_number)
    balanced_indices, cells_per_class = balanced_training_indices(train_indices, labels, class_labels, generator)

    for class_label in class_labels:
        training_records.append(
            {
                "fold": fold_number,
                "cell_line": class_label,
                "available_training_cells": int((labels[train_indices] == class_label).sum()),
                "balanced_training_cells": cells_per_class,
                "held_out_cells": int((labels[test_indices] == class_label).sum()),
                "held_out_groups": int(len(set(groups[test_indices][labels[test_indices] == class_label]))),
            }
        )

    for model_name, model in make_models(RANDOM_SEED + fold_number).items():
        model.fit(features[balanced_indices], labels[balanced_indices])
        predictions = model.predict(features[test_indices])

        overall_metrics, per_class_metrics = calculate_metrics(labels[test_indices], predictions, class_labels)

        fold_metrics.append(
            {
                "fold": fold_number,
                "model": model_name,
                "training_cells": len(balanced_indices),
                "held_out_cells": len(test_indices),
                "held_out_groups": len(test_groups),
                **overall_metrics,
            }
        )

        per_class_metrics["fold"] = fold_number
        per_class_metrics["model"] = model_name
        class_metrics.append(per_class_metrics)

        prediction_tables.append(
            create_prediction_table(cells, test_indices, fold_number, model_name, model, class_labels, predictions)
        )

        if model_name == "random_forest":
            feature_importance_tables.append(
                pd.DataFrame(
                    {"fold": fold_number, "feature": FEATURE_COLUMNS, "importance": model.feature_importances_}
                )
            )

# ---------------------------------------------------------------------
# Combine cross-validation results and select the preferred model
# ---------------------------------------------------------------------

fold_metrics_df = pd.DataFrame(fold_metrics)
class_metrics_df = pd.concat(class_metrics, ignore_index=True)
predictions_df = pd.concat(prediction_tables, ignore_index=True)
feature_importance_df = pd.concat(feature_importance_tables, ignore_index=True)
training_summary_df = pd.DataFrame(training_records)
metric_columns = ["accuracy", "balanced_accuracy", "macro_f1"]
model_summary_df = fold_metrics_df.groupby("model")[metric_columns].agg(["mean", "std"])
model_summary_df.columns = [f"{metric}_{statistic}" for metric, statistic in model_summary_df.columns]
model_summary_df = model_summary_df.reset_index()
best_model_name = model_summary_df.loc[model_summary_df["macro_f1_mean"].idxmax(), "model"]
best_predictions_df = predictions_df.loc[predictions_df["model"] == best_model_name].copy()

best_overall_metrics, best_class_metrics = calculate_metrics(
    best_predictions_df["true_cell_line"], best_predictions_df["predicted_cell_line"], class_labels
)

best_class_metrics["model"] = best_model_name
best_class_metrics["evaluation"] = "all_out_of_fold_predictions"

confusion_counts = confusion_matrix(
    best_predictions_df["true_cell_line"], best_predictions_df["predicted_cell_line"], labels=class_labels
)

confusion_normalised = np.divide(
    confusion_counts,
    confusion_counts.sum(axis=1, keepdims=True),
    out=np.zeros_like(confusion_counts, dtype=float),
    where=confusion_counts.sum(axis=1, keepdims=True) > 0,
)

confusion_count_df = pd.DataFrame(confusion_counts, index=class_labels, columns=class_labels)
confusion_normalised_df = pd.DataFrame(confusion_normalised, index=class_labels, columns=class_labels)
feature_importance_summary_df = (
    feature_importance_df.groupby("feature")
    .agg(mean_importance=("importance", "mean"), std_importance=("importance", "std"))
    .sort_values("mean_importance", ascending=False)
    .reset_index()
)

# ---------------------------------------------------------------------
# Train and save a final balanced model for future use
# ---------------------------------------------------------------------

all_indices = np.arange(len(cells))
final_generator = np.random.default_rng(RANDOM_SEED)
final_training_indices, final_cells_per_class = balanced_training_indices(
    all_indices, labels, class_labels, final_generator
)

final_model = make_models(RANDOM_SEED)[best_model_name]
final_model.fit(features[final_training_indices], labels[final_training_indices])

final_training_cells_df = cells.iloc[final_training_indices][
    ["input_cell_index", "image", "split_group", TARGET_COLUMN]
].copy()

model_output_path = OUTPUT_DIR / "multicell_line_final_classifier.joblib"
joblib.dump(
    {
        "model": final_model,
        "model_name": best_model_name,
        "feature_columns": FEATURE_COLUMNS,
        "class_labels": class_labels.tolist(),
        "training_cells_per_class": int(final_cells_per_class),
        "random_seed": RANDOM_SEED,
    },
    model_output_path,
)

# ---------------------------------------------------------------------
# Save tables and analysis manifest
# ---------------------------------------------------------------------

fold_metrics_output_path = OUTPUT_DIR / "classifier_fold_metrics.csv"
model_summary_output_path = OUTPUT_DIR / "classifier_model_summary.csv"
class_metrics_output_path = OUTPUT_DIR / "classifier_class_metrics_by_fold.csv"
best_class_metrics_output_path = OUTPUT_DIR / "classifier_best_model_class_metrics.csv"
predictions_output_path = OUTPUT_DIR / "classifier_out_of_fold_predictions.csv"
training_output_path = OUTPUT_DIR / "classifier_fold_training_summary.csv"
final_training_output_path = OUTPUT_DIR / "classifier_final_training_cells.csv"
feature_importance_output_path = OUTPUT_DIR / "classifier_random_forest_feature_importance.csv"
confusion_count_output_path = OUTPUT_DIR / "classifier_confusion_counts.csv"
confusion_normalised_output_path = OUTPUT_DIR / "classifier_confusion_normalised.csv"
manifest_output_path = OUTPUT_DIR / "classifier_manifest.json"

fold_metrics_df.to_csv(fold_metrics_output_path, index=False)
model_summary_df.to_csv(model_summary_output_path, index=False)
class_metrics_df.to_csv(class_metrics_output_path, index=False)
best_class_metrics.to_csv(best_class_metrics_output_path, index=False)
predictions_df.to_csv(predictions_output_path, index=False)
training_summary_df.to_csv(training_output_path, index=False)
final_training_cells_df.to_csv(final_training_output_path, index=False)
feature_importance_summary_df.to_csv(feature_importance_output_path, index=False)
confusion_count_df.to_csv(confusion_count_output_path, index_label="true_cell_line")
confusion_normalised_df.to_csv(confusion_normalised_output_path, index_label="true_cell_line")

manifest = {
    "analysis": "Group-aware multi-cell-line morphology classifier",
    "input_file": str(INPUT_FILE),
    "input_cells": int(len(cells)),
    "cell_lines": class_labels.tolist(),
    "feature_columns": FEATURE_COLUMNS,
    "excluded_features": {
        "raw_intensity": ("Excluded to reduce acquisition and illumination confounding."),
        "equivalent_diameter_area": "Redundant with area.",
        "eccentricity": "Overlaps with aspect ratio and circularity.",
    },
    "grouping_strategy": grouping_strategy,
    "groups_per_cell_line": {str(cell_line): int(count) for cell_line, count in group_counts.items()},
    "cross_validation_folds": int(cv_folds),
    "training_balance": (
        "Each training fold was downsampled to equal cells per line; " "all held-out cells were scored."
    ),
    "max_training_cells_per_class": MAX_TRAINING_CELLS_PER_CLASS,
    "models_compared": model_summary_df["model"].tolist(),
    "selected_model": best_model_name,
    "out_of_fold_best_model_metrics": {key: float(value) for key, value in best_overall_metrics.items()},
    "interpretation": (
        "Performance estimates are group-aware but reflect the selected "
        "LIVECell fields. They should not be treated as a diagnostic or as "
        "evidence that morphology uniquely determines biological identity."
    ),
}

with manifest_output_path.open("w", encoding="utf-8") as output_file:
    json.dump(manifest, output_file, indent=4)

# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

print("\n")
print("=" * 72)
print("MULTI-CELL-LINE PHENOTYPE CLASSIFIER")
print("=" * 72)
print(f"Input QC-passing cells:     {len(cells)}")
print(f"Cell-line classes:          {len(class_labels)}")
print(f"Group split strategy:       {grouping_strategy}")
print(f"Cross-validation folds:     {cv_folds}")
print(f"Selected model:             {best_model_name}")
print("Out-of-fold balanced accuracy: " f"{best_overall_metrics['balanced_accuracy']:.3f}")
print("Out-of-fold macro F1:        " f"{best_overall_metrics['macro_f1']:.3f}")

print("\nCross-validation model comparison:")
print(model_summary_df.round(3).to_string(index=False))

print("\nBest-model per-class performance:")
print(best_class_metrics.round(3).to_string(index=False))

print(f"\nSaved out-of-fold predictions to:\n{predictions_output_path}")
print(f"\nSaved final classifier model to:\n{model_output_path}")
print(f"\nSaved feature importance summary to:\n{feature_importance_output_path}")

# ---------------------------------------------------------------------
# Plot: model comparison, confusion, feature importance, data balance
# ---------------------------------------------------------------------

figure, axes = plt.subplots(2, 2, figsize=(14, 11))

score_columns = ["balanced_accuracy_mean", "macro_f1_mean"]
score_labels = ["Balanced accuracy", "Macro F1"]
model_positions = np.arange(len(model_summary_df))

for offset, (column, label) in zip([-0.18, 0.18], zip(score_columns, score_labels)):
    axes[0, 0].bar(model_positions + offset, model_summary_df[column], width=0.36, label=label)

axes[0, 0].set_xticks(model_positions, model_summary_df["model"])
axes[0, 0].set_ylim(0, 1)
axes[0, 0].set_ylabel("Mean cross-validation score")
axes[0, 0].set_title("Group-aware model comparison")
axes[0, 0].legend()

confusion_image = plot_confusion_matrix(
    axes[0, 1], confusion_normalised, class_labels, f"{best_model_name}: out-of-fold confusion matrix"
)
figure.colorbar(confusion_image, ax=axes[0, 1], fraction=0.046, pad=0.04, label="Fraction of true cell line")

importance_plot_data = feature_importance_summary_df.sort_values("mean_importance")
axes[1, 0].barh(
    importance_plot_data["feature"],
    importance_plot_data["mean_importance"],
    xerr=importance_plot_data["std_importance"].fillna(0),
    color="tab:green",
)
axes[1, 0].set_xlabel("Mean random-forest feature importance")
axes[1, 0].set_title("Feature importance across validation folds")

input_counts = cells.groupby(TARGET_COLUMN).size().reindex(class_labels)
final_counts = final_training_cells_df.groupby(TARGET_COLUMN).size().reindex(class_labels)
class_positions = np.arange(len(class_labels))
axes[1, 1].bar(class_positions - 0.18, input_counts, width=0.36, label="Available QC cells")
axes[1, 1].bar(class_positions + 0.18, final_counts, width=0.36, label="Final balanced training cells")
axes[1, 1].set_xticks(class_positions, class_labels)
axes[1, 1].set_ylabel("Cells")
axes[1, 1].set_title("Class balance for final model")
axes[1, 1].legend()

figure.suptitle("Multi-cell-line morphology classifier: group-aware evaluation")
plt.tight_layout()

figure_path = OUTPUT_DIR / "multicell_line_classifier_overview.png"
plt.savefig(figure_path, dpi=200, bbox_inches="tight")