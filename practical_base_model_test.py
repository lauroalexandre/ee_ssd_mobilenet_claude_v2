import torch
import torchvision
from torchvision import transforms
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
import time
import logging
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Configuration
INPUT_IMAGES_PATH = "./input_images/"
OUTPUT_BASE_PATH = "./output_images_baseline/"
PROCESSED_IMAGES_PATH = os.path.join(OUTPUT_BASE_PATH, "processed_images")
CONFIDENCE_THRESHOLD = 0.5
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp']

# Create output directories
os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)
os.makedirs(PROCESSED_IMAGES_PATH, exist_ok=True)

# COCO class names (from ssdlite_baseline.py)
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Generate distinct colors for each class
np.random.seed(42)  # For reproducible colors
CLASS_COLORS = {}
for i in range(len(COCO_CLASSES)):
    CLASS_COLORS[i] = tuple(np.random.randint(0, 255, 3).tolist())


def load_model():
    """Load pretrained SSDLite320 MobileNet V3 Large model """
    logger.info("Loading SSDLite320 MobileNet V3 Large model...")
    start_time = time.time()

    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
        weights=torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
    )
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    load_time = time.time() - start_time
    logger.info(f"Model loaded successfully on device: {device} in {load_time:.2f}s")

    return model, device


def preprocess_image_for_inference(image_path):
    """Preprocess image for inference """
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor()
    ])
    return transform(image)


def preprocess_image_for_visualization(image_path):
    """Load original image for visualization with OpenCV"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def draw_detections(image, predictions, original_size, confidence_threshold=CONFIDENCE_THRESHOLD):
    """Draw bounding boxes and labels on image with proper coordinate scaling"""
    image_copy = image.copy()
    original_height, original_width = original_size

    boxes = predictions['boxes'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()

    # Filter by confidence threshold
    valid_detections = scores >= confidence_threshold
    boxes = boxes[valid_detections]
    labels = labels[valid_detections]
    scores = scores[valid_detections]

    # Calculate scaling factors from 320x320 to original image size
    scale_x = original_width / 320.0
    scale_y = original_height / 320.0

    for box, label, score in zip(boxes, labels, scores):
        # Scale coordinates back to original image size
        x1 = int(box[0] * scale_x)
        y1 = int(box[1] * scale_y)
        x2 = int(box[2] * scale_x)
        y2 = int(box[3] * scale_y)

        # Clamp coordinates to image boundaries
        x1 = max(0, min(x1, original_width - 1))
        y1 = max(0, min(y1, original_height - 1))
        x2 = max(0, min(x2, original_width - 1))
        y2 = max(0, min(y2, original_height - 1))

        # Get class name and color
        class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f'Class_{label}'
        color = CLASS_COLORS.get(label, (255, 255, 255))

        # Thickness proportional to confidence (1-5 pixels)
        thickness = max(1, int(score * 5))

        # Draw bounding box
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, thickness)

        # Prepare label text
        label_text = f'{class_name}: {score:.2f}'

        # Calculate text size and background
        font_scale = max(0.4, min(0.8, (x2 - x1) / 150.0))  # Adaptive font size
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
        )

        # Draw label background
        label_y = y1 - 10 if y1 - 10 > text_height else y1 + text_height + 15
        cv2.rectangle(
            image_copy,
            (x1, label_y - text_height - 5),
            (x1 + text_width + 5, label_y + 5),
            color,
            -1
        )

        # Draw label text
        text_color = (0, 0, 0) if sum(color) > 400 else (255, 255, 255)
        cv2.putText(
            image_copy,
            label_text,
            (x1 + 2, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            2
        )

    return image_copy, len(boxes)


def process_single_image(model, device, image_path):
    """Process a single image with timing and detailed metrics collection"""
    filename = os.path.basename(image_path)

    try:
        # Load images
        original_image = preprocess_image_for_visualization(image_path)
        height, width = original_image.shape[:2]

        # Preprocess for inference
        input_tensor = preprocess_image_for_inference(image_path).unsqueeze(0).to(device)

        # Run inference with timing
        with torch.no_grad():
            start_time = time.time()
            predictions = model(input_tensor)
            inference_time_ms = (time.time() - start_time) * 1000

        # Extract prediction data
        prediction = predictions[0]
        boxes = prediction['boxes'].cpu().numpy()
        labels = prediction['labels'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()

        # Filter valid detections
        valid_detections = scores >= CONFIDENCE_THRESHOLD
        valid_scores = scores[valid_detections]
        valid_labels = labels[valid_detections]

        # Calculate metrics
        total_objects = len(valid_scores)
        avg_confidence = float(np.mean(valid_scores)) if len(valid_scores) > 0 else 0.0

        # Count classes detected
        class_counts = defaultdict(int)
        for label in valid_labels:
            class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f'Class_{label}'
            class_counts[class_name] += 1

        classes_detected = ','.join([f'{cls}:{count}' for cls, count in class_counts.items()]) if class_counts else ''

        # Draw detections and save image (pass original image dimensions)
        annotated_image, num_drawn = draw_detections(original_image, prediction, (height, width))

        # Save processed image
        output_filename = f"processed_{filename}"
        output_path = os.path.join(PROCESSED_IMAGES_PATH, output_filename)

        # Convert RGB back to BGR for OpenCV saving
        annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, annotated_image_bgr)

        # Collect metrics
        metrics = {
            'filename': filename,
            'width': width,
            'height': height,
            'inference_time_ms': round(inference_time_ms, 2),
            'total_objects': total_objects,
            'avg_confidence': round(avg_confidence, 4),
            'classes_detected': classes_detected,
            'processing_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        logger.info(f"Processed {filename}: {total_objects} objects in {inference_time_ms:.1f}ms")
        return metrics

    except Exception as e:
        logger.error(f"Error processing {filename}: {str(e)}")
        return {
            'filename': filename,
            'width': 0,
            'height': 0,
            'inference_time_ms': 0,
            'total_objects': 0,
            'avg_confidence': 0.0,
            'classes_detected': '',
            'processing_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'error': str(e)
        }


def save_csv_report(metrics_list):
    """Save detailed metrics to CSV report"""
    csv_path = os.path.join(OUTPUT_BASE_PATH, "inference_report.csv")

    df = pd.DataFrame(metrics_list)

    # Ensure column order matches specification
    column_order = [
        'filename', 'width', 'height', 'inference_time_ms',
        'total_objects', 'avg_confidence', 'classes_detected', 'processing_timestamp'
    ]

    # Add error column if it exists
    if 'error' in df.columns:
        column_order.append('error')

    df = df[column_order]
    df.to_csv(csv_path, index=False)

    logger.info(f"CSV report saved to: {csv_path}")
    return csv_path


def generate_performance_chart(metrics_list):
    """Generate comprehensive performance analysis chart"""
    chart_path = os.path.join(OUTPUT_BASE_PATH, "performance_chart.png")

    # Extract data for plotting
    filenames = [m['filename'] for m in metrics_list if 'error' not in m]
    inference_times = [m['inference_time_ms'] for m in metrics_list if 'error' not in m]
    object_counts = [m['total_objects'] for m in metrics_list if 'error' not in m]
    confidences = [m['avg_confidence'] for m in metrics_list if 'error' not in m and m['avg_confidence'] > 0]

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('SSDLite320 MobileNet V3 Large - Performance Analysis', fontsize=16, fontweight='bold')

    # 1. Inference time per image (bar chart)
    bars = ax1.bar(range(len(filenames)), inference_times, color='steelblue', alpha=0.7)
    ax1.set_title('Inference Time per Image', fontweight='bold')
    ax1.set_xlabel('Images')
    ax1.set_ylabel('Inference Time (ms)')
    ax1.set_xticks(range(len(filenames)))
    ax1.set_xticklabels([f.split('.')[0] for f in filenames], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add trend line
    if len(inference_times) > 1:
        avg_time = np.mean(inference_times)
        ax1.axhline(y=avg_time, color='red', linestyle='--', alpha=0.8,
                   label=f'Average: {avg_time:.1f}ms')
        ax1.legend()

    # Add value labels on bars
    for bar, time_val in zip(bars, inference_times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(inference_times)*0.01,
                f'{time_val:.1f}', ha='center', va='bottom', fontsize=9)

    # 2. Inference time distribution (histogram)
    ax2.hist(inference_times, bins=min(10, len(inference_times)), alpha=0.7, color='green', edgecolor='black')
    ax2.set_title('Inference Time Distribution', fontweight='bold')
    ax2.set_xlabel('Inference Time (ms)')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)

    # Add statistics text
    if inference_times:
        stats_text = f'Min: {min(inference_times):.1f}ms\n'
        stats_text += f'Max: {max(inference_times):.1f}ms\n'
        stats_text += f'Mean: {np.mean(inference_times):.1f}ms\n'
        stats_text += f'Std: {np.std(inference_times):.1f}ms'
        ax2.text(0.98, 0.95, stats_text, transform=ax2.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 3. Objects detected per image
    bars3 = ax3.bar(range(len(filenames)), object_counts, color='orange', alpha=0.7)
    ax3.set_title('Objects Detected per Image', fontweight='bold')
    ax3.set_xlabel('Images')
    ax3.set_ylabel('Number of Objects')
    ax3.set_xticks(range(len(filenames)))
    ax3.set_xticklabels([f.split('.')[0] for f in filenames], rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, count in zip(bars3, object_counts):
        height = bar.get_height()
        if height > 0:
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{count}', ha='center', va='bottom', fontsize=9)

    # 4. Inference time vs Objects detected (scatter plot)
    if inference_times and object_counts:
        scatter = ax4.scatter(object_counts, inference_times, alpha=0.7, s=60, c='red')
        ax4.set_title('Inference Time vs Objects Detected', fontweight='bold')
        ax4.set_xlabel('Number of Objects Detected')
        ax4.set_ylabel('Inference Time (ms)')
        ax4.grid(True, alpha=0.3)

        # Add correlation coefficient if possible
        if len(object_counts) > 1 and len(inference_times) > 1:
            correlation = np.corrcoef(object_counts, inference_times)[0, 1]
            ax4.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                    transform=ax4.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        # Add trend line
        if len(object_counts) > 1:
            z = np.polyfit(object_counts, inference_times, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(object_counts), max(object_counts), 100)
            ax4.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)

    plt.tight_layout()
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Performance chart saved to: {chart_path}")
    return chart_path


def main():
    """Main function to run the complete inference pipeline"""
    logger.info("Starting SSDLite320 MobileNet V3 Large baseline inference test...")

    # Validate input directory
    if not os.path.exists(INPUT_IMAGES_PATH):
        logger.error(f"Input directory not found: {INPUT_IMAGES_PATH}")
        return

    # Find all supported images
    image_files = []
    for filename in os.listdir(INPUT_IMAGES_PATH):
        file_ext = os.path.splitext(filename.lower())[1]
        if file_ext in SUPPORTED_FORMATS:
            image_files.append(os.path.join(INPUT_IMAGES_PATH, filename))

    if not image_files:
        logger.error(f"No supported image files found in {INPUT_IMAGES_PATH}")
        logger.info(f"Supported formats: {', '.join(SUPPORTED_FORMATS)}")
        return

    logger.info(f"Found {len(image_files)} images in {INPUT_IMAGES_PATH}")

    # Load model
    try:
        model, device = load_model()
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return

    # Perform warm-up inference
    logger.info("Performing warm-up inference...")
    try:
        warmup_tensor = preprocess_image_for_inference(image_files[0]).unsqueeze(0).to(device)
        with torch.no_grad():
            _ = model(warmup_tensor)
        logger.info("Warm-up completed, starting actual inference...")
    except Exception as e:
        logger.warning(f"Warm-up failed: {str(e)}")

    # Process all images
    all_metrics = []
    logger.info(f"Processing images with confidence threshold: {CONFIDENCE_THRESHOLD}")

    for image_path in tqdm(image_files, desc="Processing images"):
        metrics = process_single_image(model, device, image_path)
        all_metrics.append(metrics)

    # Calculate summary statistics
    successful_metrics = [m for m in all_metrics if 'error' not in m]
    if successful_metrics:
        inference_times = [m['inference_time_ms'] for m in successful_metrics]
        total_objects = sum(m['total_objects'] for m in successful_metrics)

        avg_inference_time = np.mean(inference_times)
        min_inference_time = min(inference_times)
        max_inference_time = max(inference_times)
        std_inference_time = np.std(inference_times)

        logger.info(f"\n{'='*60}")
        logger.info("INFERENCE COMPLETED - SUMMARY STATISTICS")
        logger.info(f"{'='*60}")
        logger.info(f"Total images processed: {len(successful_metrics)}")
        logger.info(f"Average inference time: {avg_inference_time:.1f}ms")
        logger.info(f"Min inference time: {min_inference_time:.1f}ms")
        logger.info(f"Max inference time: {max_inference_time:.1f}ms")
        logger.info(f"Std deviation: {std_inference_time:.1f}ms")
        logger.info(f"Total objects detected: {total_objects}")
        logger.info(f"Average objects per image: {total_objects/len(successful_metrics):.1f}")
        logger.info(f"{'='*60}")

    # Handle errors
    error_count = len(all_metrics) - len(successful_metrics)
    if error_count > 0:
        logger.warning(f"Failed to process {error_count} images")

    # Generate reports
    logger.info("Generating reports...")

    # Save CSV report
    csv_path = save_csv_report(all_metrics)

    # Generate performance chart
    if successful_metrics:
        chart_path = generate_performance_chart(successful_metrics)
    else:
        logger.warning("No successful inferences to generate performance chart")
        chart_path = None

    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info("PROCESSING COMPLETED")
    logger.info(f"{'='*60}")
    logger.info(f"Processed images saved to: {PROCESSED_IMAGES_PATH}")
    logger.info(f"CSV report saved to: {csv_path}")
    if chart_path:
        logger.info(f"Performance chart saved to: {chart_path}")
    logger.info(f"Results directory: {OUTPUT_BASE_PATH}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()