"""Run baseline Cellpose segmentation and evaluate it on an A172 LIVECell image.

The script segments one annotated image with the Cellpose CPSAM model, saves
the predicted mask, visualises the result, and prepares the annotations used
for the accompanying segmentation evaluation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from cellpose import models
from pycocotools.coco import COCO

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path().resolve().parent

IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "images" / "livecell_train_val_images"
ANNOTATION_FILE = PROJECT_ROOT / "data" / "annotations" / "a172_train.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

# ---------------------------------------------------------------------
# Load up image
# ---------------------------------------------------------------------

coco = COCO(str(ANNOTATION_FILE))

image_id = next(iter(coco.imgs))
image_metadata = coco.imgs[image_id]
filename = image_metadata["file_name"]
image_path = IMAGE_DIR / filename

image = tifffile.imread(image_path)

print(f"Image: {filename}")
print(f"Shape: {image.shape}")

# ---------------------------------------------------------------------
# Load Cellpose model
# ---------------------------------------------------------------------

model = models.CellposeModel(gpu=True, pretrained_model="cpsam_v2")

# ---------------------------------------------------------------------
# Run segmentation
# ---------------------------------------------------------------------

print("Running Cellpose...")

masks, flows, styles = model.eval(image, channel_axis=None, normalize=True)

print("Cellpose finished.")
print(f"Predicted cells: {len(np.unique(masks)) - 1}")

# ---------------------------------------------------------------------
# Save predicted masks
# ---------------------------------------------------------------------

output_path = OUTPUT_DIR / f"{Path(filename).stem}_cellpose_masks.tif"

tifffile.imwrite(output_path, masks.astype(np.uint16))

print(f"Saved masks to:\n{output_path}")

# ---------------------------------------------------------------------
# Visualise the prediction
# ---------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 7))

ax.imshow(image, cmap="gray")

# Draw Cellpose boundaries
ax.contour(masks, levels=np.arange(1, masks.max() + 1), linewidths=0.5)

ax.set_title(f"Cellpose prediction\n" f"{filename}\n" f"{len(np.unique(masks)) - 1} predicted cells")

ax.axis("off")

plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------
# Evaluate cellpose
# ---------------------------------------------------------------------

# Reload predictions
MASK_FILE = PROJECT_ROOT / "data" / "processed" / "A172_Phase_D7_1_02d04h00m_3_cellpose_masks.tif"
prediction = tifffile.imread(MASK_FILE)

# Define some parameters
IOU_THRESHOLD = 0.50
FILENAME = "A172_Phase_D7_1_02d04h00m_3.tif"

# Load annotations
coco = COCO(str(ANNOTATION_FILE))

# Get the image
matching_image_ids = [image_id for image_id, metadata in coco.imgs.items() if metadata["file_name"] == FILENAME]

image_id = matching_image_ids[0]

annotation_ids = coco.getAnnIds(imgIds=[image_id])
annotations = coco.loadAnns(annotation_ids)

print(f"Ground-truth cells: {len(annotations)}")

# Convert COCO annotations into an instance-label image
ground_truth = np.zeros(image.shape, dtype=np.uint16)
for label, annotation in enumerate(annotations, start=1):
    mask = coco.annToMask(annotation)
    ground_truth[mask > 0] = label

n_ground_truth = len(annotations)
print(f"Ground-truth labels: " f"{len(np.unique(ground_truth)) - 1}")