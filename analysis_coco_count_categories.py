import json
import os
import csv
import matplotlib.pyplot as plt
from collections import defaultdict

# Define COCO 2017 dataset paths
TRAIN_ANN_PATH = "D:\\Download\\JDownloader\\MSCOCO\\annotations\\instances_train2017.json"
VAL_ANN_PATH = "D:\\Download\\JDownloader\\MSCOCO\\annotations\\instances_val2017.json"
TEST_ANN_PATH = "D:\\Download\\JDownloader\\MSCOCO\\annotations\\image_info_test2017.json"

# Output folder for generated plots and CSVs
OUTPUT_DIR = "analysis_coco_count_categories"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_annotations(json_path):
    """Load COCO annotations from a JSON file."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Annotation file not found at {json_path}. Returning empty data.")
        return {}

def analyze_dataset(annotations):
    """
    Count total annotations per category, ignoring iscrowd.
    Returns a dictionary of counts and the category map.
    """
    counts = defaultdict(int)
    category_map = {cat["id"]: cat["name"] for cat in annotations.get("categories", [])}

    for ann in annotations.get("annotations", []):
        cat_id = ann["category_id"]
        cat_name = category_map.get(cat_id, "unknown")
        counts[(cat_id, cat_name)] += 1

    return counts, category_map

def save_to_csv(dataset_name, counts):
    """Save analysis results to a CSV file."""
    csv_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_analysis.csv")
    with open(csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["category_id", "name", "total_annotations"])

        # Sort by category name for consistent output
        for (cat_id, cat_name), count in sorted(counts.items(), key=lambda item: item[0][1]):
            writer.writerow([cat_id, cat_name, count])

    print(f"CSV saved: {csv_path}")

def plot_analysis(dataset_name, counts):
    """Plot analysis by category for one dataset split."""
    if not counts:
        print(f"No data to plot for {dataset_name}")
        plt.figure(figsize=(14, 6))
        plt.title(f"COCO 2017 {dataset_name}: Annotation Distribution by Category")
        plt.xlabel("Category")
        plt.ylabel("Number of Annotations")
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_analysis.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Empty plot saved for {dataset_name}: {output_path}")
        return

    sorted_items = sorted(counts.items(), key=lambda item: item[0][1])
    categories = [item[0][1] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    plt.figure(figsize=(14, 6))
    x = range(len(categories))
    plt.bar(x, values, alpha=0.7)

    plt.title(f"COCO 2017 {dataset_name}: Annotation Distribution by Category")
    plt.xlabel("Category")
    plt.ylabel("Number of Annotations")
    plt.xticks(x, categories, rotation=90)
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_analysis.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Plot saved: {output_path}")

def main():
    """Main execution routine."""
    print("Loading annotations...")

    train_ann = load_annotations(TRAIN_ANN_PATH)
    val_ann = load_annotations(VAL_ANN_PATH)
    test_ann = load_annotations(TEST_ANN_PATH)

    print("Analyzing datasets...")

    # Train
    train_counts, _ = analyze_dataset(train_ann)
    plot_analysis("train2017", train_counts)
    save_to_csv("train2017", train_counts)

    # Validation
    val_counts, _ = analyze_dataset(val_ann)
    plot_analysis("val2017", val_counts)
    save_to_csv("val2017", val_counts)

    # Test (usually unlabeled, so counts will be empty)
    test_counts, _ = analyze_dataset(test_ann)
    plot_analysis("test2017", test_counts)
    save_to_csv("test2017", test_counts)

    print("\nDone! All plots and CSV files saved in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()