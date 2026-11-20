"""
This script creates a synthetic sample of the IDD dataset with 3 images and 9 annotations.
The images are generated with simple colored rectangles representing different categories.
Usually, we do not use real images first when we are still debugging the pipeline. We use real images first only when we already know the system is working and we want to validate performance under actual conditions.

In this project, the intended flow is:

make_sample_idd.py = quick smoke test
detector.py = actual detection logic
serve/gateway.py = backend API
real traffic images = final validation on realistic data
"""

import json
from pathlib import Path

from PIL import Image, ImageDraw

root = Path(__file__).resolve().parent
image_dir = root / "data" / "images" / "idd"
ann_dir = root / "data" / "idd" / "annotations"
image_dir.mkdir(parents=True, exist_ok=True)
ann_dir.mkdir(parents=True, exist_ok=True)

categories = [
    {"id": 1, "name": "car", "supercategory": "vehicle"},
    {"id": 2, "name": "motorcycle", "supercategory": "vehicle"},
    {"id": 3, "name": "bus", "supercategory": "vehicle"},
    {"id": 4, "name": "person", "supercategory": "human"},
    {"id": 5, "name": "truck", "supercategory": "vehicle"},
]

images = [
    {"id": 1, "file_name": "image1.jpg", "width": 640, "height": 480},
    {"id": 2, "file_name": "image2.jpg", "width": 640, "height": 480},
    {"id": 3, "file_name": "image3.jpg", "width": 640, "height": 480},
]

# bbox = [x, y, width, height]
# x and y are the top-left coordinates of the bounding box
annotations = [
    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 100, 200, 150], "area": 30000, "iscrowd": 0},
    {"id": 2, "image_id": 1, "category_id": 3, "bbox": [300, 200, 50, 100], "area": 5000, "iscrowd": 0},
    {"id": 3, "image_id": 2, "category_id": 2, "bbox": [150, 150, 100, 50], "area": 5000, "iscrowd": 0},
    {"id": 4, "image_id": 2, "category_id": 4, "bbox": [400, 300, 150, 100], "area": 15000, "iscrowd": 0},
    {"id": 5, "image_id": 2, "category_id": 5, "bbox": [200, 100, 300, 200], "area": 60000, "iscrowd": 0},
    {"id": 6, "image_id": 2, "category_id": 1, "bbox": [50, 50, 100, 100], "area": 10000, "iscrowd": 0},
    {"id": 7, "image_id": 3, "category_id": 1, "bbox": [250, 200, 100, 150], "area": 15000, "iscrowd": 0},
    {"id": 8, "image_id": 3, "category_id": 2, "bbox": [400, 300, 50, 100], "area": 5000, "iscrowd": 0},
    {"id": 9, "image_id": 3, "category_id": 4, "bbox": [100, 100, 200, 150], "area": 30000, "iscrowd": 0},
]

fill = {
    1: (120, 120, 200),
    2: (120, 200, 120),
    3: (200, 160, 100),
    4: (200, 120, 120),
    5: (150, 150, 150),
}

for info in images:
    canvas = Image.new("RGB", (info["width"], info["height"]), (225, 225, 225))
    draw = ImageDraw.Draw(canvas)

    for ann in annotations:
        if ann["image_id"] == info["id"]:
            x, y, w, h = ann["bbox"]
            draw.rectangle([x, y, x + w, y + h], fill=fill[ann["category_id"]])

    canvas.save(image_dir / info["file_name"])

annotation_file = ann_dir / "instances_default.json"
with open(annotation_file, "w", encoding="utf-8") as f:
    json.dump({
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }, f, indent=4)

print("Created synthetic sample images in data/images/idd")
print(f"Saved annotation file: {annotation_file}")
print("This is synthetic data for testing purposes, not the official IDD dataset.")

