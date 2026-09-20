# Quantitative-analysis-of-cancer-cell-morphology-and-phenotype-from-live-cell-microscopy

## Quantitative bioimage analysis with LIVECell phase-contrast microscopy

This repository contains a Python workflow for moving from phase-contrast microscopy images to quantitative, single-cell analysis. It asks whether morphology and local crowding could describe changing phenotypes and distinguish several cell lines.

The analysis uses images and COCO-format annotations from the LIVECell dataset. It is not intended as a diagnostic tool or a claim that morphology alone defines cell identity. The emphasis is on a transparent, reproducible workflow with visual checks and held-out evaluation.

**What the pipeline does**

Loads phase-contrast TIFF images and COCO instance annotations. 

Converts annotations into instance-label masks and checks them against the source images.

Runs Cellpose instance segmentation and evaluates it against annotated A172 images.

Tunes Cellpose parameters on a development set and assesses the selected settings on held-out images.

Runs the selected model across A172 images, creating masks and per-cell morphology, intensity, and neighbourhood measurements.

Explores A172 morphology with quality control, PCA and clustering, then tests whether the clusters are stable under bootstrap refitting.

Extracts comparable features from A172, BT474, BV2, Huh7 and MCF7 images.

Trains a morphology-based cell-line classifier using field-aware cross-validation, so cells from the same imaging field cannot occur in both training and test data.

## Key results

**Segmentation**

Cellpose parameters were selected only using a tuning subset of annotated A172 images. On held-out A172 images, the selected configuration improved F1 from 0.785 for the baseline to 0.819. The final 250-image multi-cell-line extraction used diameter 20, cell-probability threshold 0.0, flow threshold 0.6 and a minimum object size of 15 pixels.

For the 50-image-per-line analysis, 74,670 cells were segmented and 66,349 passed classifier quality control. A172 transfer evaluation on 20 annotated images gave an F1 of 0.807 and mean matched IoU of 0.788.

**A172 phenotype exploration**

After excluding border-touching, extreme-area and incomplete observations, 952 A172 cells were retained for exploratory analysis. PCA and clustering supported three reproducible morphology-and-crowding profiles:

a compact, relatively round and crowded group;\
a compact but more isolated group\
a larger, less circular and more elongated group.

The composition changed over the recorded time course: the first profile decreased while the elongated profile increased. Bootstrap refitting gave a median adjusted Rand index of 0.891, which supports internal stability but does not turn the exploratory clusters into fixed biological cell states.

**Multi-cell-line classifier**

The final classifier was evaluated with grouped splits based on cell line, well and location. This prevents cells from a held-out field entering the corresponding training data. A random forest performed best, reaching an out-of-fold balanced accuracy of 0.716 and macro F1 of 0.683.

BV2 was the most separable line (F1 0.905). A172 and MCF7 were moderately well separated, while BT474 and Huh7 had more overlap with the remaining classes. Local-neighbourhood features, area and circularity were the most influential features in the random-forest model.

## Quality control and limitations

Segmentation was tuned and quantitatively evaluated on A172 annotations. The same parameters were deliberately transferred to the other lines to keep the comparison consistent, but their segmentation accuracy was not independently verified here because matching annotations were unavailable.

The classifier is a morphology-and-crowding classifier, not a cell-line authentication method. Differences in density, acquisition conditions or the sampled fields can contribute to its predictions.

Cell-level samples are correlated within an imaging field. Group-aware cross-validation addresses direct train/test leakage, although the number of independent fields remains smaller than the number of cells.

The A172 clusters are descriptive, internally validated groupings. Biological interpretation would need orthogonal evidence, such as markers, perturbations or time-resolved cell tracking.


## Data citation

Edlund C, Jackson TR, Khalid N et al. LIVECell—A large-scale dataset for label-free live cell segmentation. Nature Methods (2021)
