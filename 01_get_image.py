"""Inspect one annotated A172 LIVECell image and derive cell measurements.

The script loads a selected image and its COCO annotations, visualises cell
boundaries, creates an instance-label image, and reports basic morphology
measurements for the annotated cells.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib.patches import Polygon
from pycocotools.coco import COCO
from skimage.measure import regionprops_table

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path().resolve().parent
IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "images" / "livecell_train_val_images"
ANNOTATION_FILE = PROJECT_ROOT / "data" / "annotations" / "a172_train.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
images = sorted(IMAGE_DIR.glob("*.tif"))

# ---------------------------------------------------------------------
# Load COCO annotations
# ---------------------------------------------------------------------

coco = COCO(str(ANNOTATION_FILE))

print(f"Images in annotation file: {len(coco.imgs)}")
print(f"Annotations in annotation file: {len(coco.anns)}")

# ---------------------------------------------------------------------
# Select an image from the annotation file
# ---------------------------------------------------------------------

image_id = next(iter(coco.imgs))
image_metadata = coco.imgs[image_id]
filename = image_metadata["file_name"]
print(f"Selected image: {filename}")

# Find the actual image on disk
image_path = IMAGE_DIR / filename
print(f"Image path: {image_path}")

# Load image
image = tifffile.imread(image_path)
print(f"Image shape: {image.shape}")
print(f"Image dtype: {image.dtype}")

# Get annotations for this image
annotation_ids = coco.getAnnIds(imgIds=[image_id])
annotations = coco.loadAnns(annotation_ids)
print(f"Cells annotated: {len(annotations)}")

# ---------------------------------------------------------------------
# Display image and cell boundaries
# ---------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 7))

ax.imshow(image, cmap="gray")

for annotation in annotations:
    segmentation = annotation["segmentation"]

    for polygon in segmentation:
        points = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
        patch = Polygon(points, closed=True, fill=False, linewidth=1.0)

        ax.add_patch(patch)

ax.set_title(f"{filename}\n" f"{len(annotations)} annotated cells")

ax.axis("off")

plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------
# Create instance label image
# ---------------------------------------------------------------------

label_image = np.zeros(image.shape, dtype=np.uint16)

for cell_id, annotation in enumerate(annotations, start=1):
    mask = coco.annToMask(annotation)
    label_image[mask > 0] = cell_id

print(f"Background pixels: {(label_image == 0).sum()}")
print(f"Cell pixels:       {(label_image > 0).sum()}")
print(f"Unique labels:     {len(np.unique(label_image))}")

# Save label image
output_path = OUTPUT_DIR / f"{Path(filename).stem}_labels.tif"

tifffile.imwrite(output_path, label_image)
print(f"Saved label image to:\n{output_path}")

# Display label image
plt.figure(figsize=(10, 7))
plt.imshow(label_image, cmap="nipy_spectral")
plt.title(f"Instance labels\n" f"{filename}\n" f"{len(annotations)} cells")
plt.axis("off")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------
# Extract cell-level measurements
# ---------------------------------------------------------------------

properties = ["label", "area", "perimeter", "eccentricity", "solidity", "extent", "centroid"]
measurements = regionprops_table(label_image, intensity_image=image, properties=properties)
cells = pd.DataFrame(measurements)

print("\nFirst five cells:")
print(cells.head())

print("\nSummary:")
print(cells.describe())
