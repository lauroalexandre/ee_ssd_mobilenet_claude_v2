import json
import os
import shutil
import random
import argparse

# -----------------------------
# Argument configuration
# -----------------------------
parser = argparse.ArgumentParser(description="Generate a COCO dataset subset for validating an early-exit model.")
parser.add_argument("--category", required=True, help="COCO category name (e.g., chair, person, car, etc.)")
parser.add_argument("--val_images_path", default=r"D:\Download\JDownloader\MSCOCO\images\val2017", help="Path to COCO validation images")
parser.add_argument("--val_annotations_path", default=r"D:\Download\JDownloader\MSCOCO\annotations\instances_val2017.json", help="Path to COCO annotation JSON file")
parser.add_argument("--output_dir", default="subset_validation", help="Base output directory")
parser.add_argument("--limit", type=int, default=180, help="Maximum number of images per group (single/crowd)")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

args = parser.parse_args()

# -----------------------------
# Set random seed
# -----------------------------
random.seed(args.seed)

# -----------------------------
# Paths and parameters
# -----------------------------
CATEGORY_NAME = args.category.lower()
OUTPUT_DIR = os.path.join(args.output_dir, f"subset_{CATEGORY_NAME}_validation")
SINGLE_DIR = os.path.join(OUTPUT_DIR, "single")
CROWD_DIR = os.path.join(OUTPUT_DIR, "crowd")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, f"{CATEGORY_NAME}_coco2017_subset.json")

# Create output folders
os.makedirs(SINGLE_DIR, exist_ok=True)
os.makedirs(CROWD_DIR, exist_ok=True)

# -----------------------------
# Load COCO dataset
# -----------------------------
with open(args.val_annotations_path, "r", encoding="utf-8") as f:
    coco_data = json.load(f)

# Find category ID
category = next((c for c in coco_data["categories"] if c["name"].lower() == CATEGORY_NAME), None)
if not category:
    raise ValueError(f"Category '{CATEGORY_NAME}' not found in COCO dataset.")

category_id = category["id"]

# -----------------------------
# Filter annotations by category
# -----------------------------
annotations = [ann for ann in coco_data["annotations"] if ann["category_id"] == category_id]
annotations_single = [ann for ann in annotations if not ann.get("iscrowd", 0)]
annotations_crowd = [ann for ann in annotations if ann.get("iscrowd", 0)]

# Shuffle and limit crowd annotations
random.shuffle(annotations_crowd)
annotations_crowd = annotations_crowd[:args.limit]

# Get image IDs from the crowd subset to avoid duplicates
crowd_image_ids = {ann["image_id"] for ann in annotations_crowd}

# Filter single annotations to exclude images already in the crowd subset
annotations_single = [ann for ann in annotations_single if ann["image_id"] not in crowd_image_ids]

# Shuffle and limit single annotations
random.shuffle(annotations_single)
annotations_single = annotations_single[:args.limit]

# Map images (id -> info)
images_dict = {img["id"]: img for img in coco_data["images"]}

# -----------------------------
# Helper function to process subsets
# -----------------------------
def process_subset(annotations, dest_folder):
    subset_data = []
    for ann in annotations:
        img_info = images_dict[ann["image_id"]]
        src_path = os.path.join(args.val_images_path, img_info["file_name"])
        dst_path = os.path.join(dest_folder, img_info["file_name"])

        if not os.path.exists(src_path):
            print(f"Image not found: {src_path}")
            continue

        # Copy image (without overwriting)
        if not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)

        subset_data.append({
            "id": ann["image_id"],
            "file_name": img_info["file_name"],
            "category_id": ann["category_id"],
            "iscrowd": ann["iscrowd"],
            "bbox": ann["bbox"],
            "path": os.path.relpath(dst_path, start=OUTPUT_DIR)
        })
    return subset_data

# -----------------------------
# Process both subsets
# -----------------------------
subset_single = process_subset(annotations_single, SINGLE_DIR)
subset_crowd = process_subset(annotations_crowd, CROWD_DIR)

# Merge and save final JSON file
subset_all = subset_single + subset_crowd
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(subset_all, f, indent=4, ensure_ascii=False)

# -----------------------------
# Summary
# -----------------------------
print(f"\nSubset generated for category: '{CATEGORY_NAME}'")
print(f"Base directory: {OUTPUT_DIR}")
print(f"Single images: {len(subset_single)}")
print(f"Crowd images: {len(subset_crowd)}")
print(f"Limit per group: {args.limit}")
print(f"Random seed: {args.seed}")
print(f"JSON file: {OUTPUT_JSON}")
