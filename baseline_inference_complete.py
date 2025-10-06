#!/usr/bin/env python3
"""
SSDLite320 MobileNetV3 Large - Baseline Inference System
Complete implementation for object detection baseline evaluation

Author: Baseline Inference System
Date: October 2025
Version: 1.0

Usage:
    python baseline_inference_complete.py

Requirements:
    - torch
    - torchvision
    - pillow
    - matplotlib
    - numpy
"""

import torch
import torchvision
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw, ImageFont
import json
import os
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
    'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A',
    'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_header(text: str, char: str = "=", width: int = 80):
    """Print formatted header"""
    print()
    print(char * width)
    print(text)
    print(char * width)


def print_section(text: str, width: int = 80):
    """Print formatted section"""
    print()
    print("-" * width)
    print(text)
    print("-" * width)


def check_environment():
    """Check and display environment information"""
    print_header("ENVIRONMENT CHECK")

    print(f"Python version: {torch.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"TorchVision version: {torchvision.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("Running in CPU-only mode")


def create_output_directories(base_path: str):
    """Create necessary output directories"""
    paths = [
        Path(base_path),
        Path(base_path) / "images_baseline",
    ]

    for path in paths:
        path.mkdir(parents=True, exist_ok=True)

    print(f"Output directories created at: {base_path}")


def get_image_files(folder: str) -> List[Path]:
    """Get all image files from folder"""
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
    image_files = []

    folder_path = Path(folder)
    for ext in extensions:
        image_files.extend(folder_path.glob(f'*{ext}'))

    # Remove duplicates (case-insensitive file systems return duplicates)
    unique_files = list(dict.fromkeys(image_files))

    return sorted(unique_files)


# ============================================================================
# INFERENCE ENGINE
# ============================================================================

class BaselineInferenceEngine:
    """
    Complete inference engine for baseline object detection
    """

    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize inference engine

        Args:
            confidence_threshold: Minimum confidence score for detections
        """
        self.confidence_threshold = confidence_threshold
        self.device_cpu = torch.device('cpu')
        self.device_gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_cpu = None
        self.model_gpu = None

        print(f"Inference engine initialized (confidence threshold: {confidence_threshold})")

    def load_model(self, use_gpu: bool = False) -> torch.nn.Module:
        """
        Load pretrained SSDLite320 MobileNetV3 Large model

        Args:
            use_gpu: Whether to load model on GPU

        Returns:
            Loaded model
        """
        device = self.device_gpu if use_gpu else self.device_cpu
        device_name = "GPU" if use_gpu else "CPU"

        print(f"Loading model on {device_name}...", end=" ")

        model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
            weights=torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
        )

        model.to(device)
        model.eval()

        print("Done!")

        if use_gpu:
            self.model_gpu = model
        else:
            self.model_cpu = model

        return model

    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, Image.Image]:
        """
        Load and preprocess image

        Args:
            image_path: Path to image file

        Returns:
            Tuple of (tensor, PIL Image)
        """
        image = Image.open(image_path).convert('RGB')
        image_tensor = F.to_tensor(image)
        return image_tensor, image

    def run_inference(
        self, 
        image_tensor: torch.Tensor, 
        model: torch.nn.Module, 
        device: torch.device
    ) -> Tuple[Dict, float]:
        """
        Run inference on single image

        Args:
            image_tensor: Preprocessed image tensor
            model: Detection model
            device: Device to run on

        Returns:
            Tuple of (predictions, inference_time)
        """
        image_tensor = image_tensor.to(device)

        # Warm-up for GPU
        if device.type == 'cuda':
            with torch.no_grad():
                _ = model([image_tensor])
            torch.cuda.synchronize()

        # Measure inference time
        start_time = time.perf_counter()

        with torch.no_grad():
            predictions = model([image_tensor])

        if device.type == 'cuda':
            torch.cuda.synchronize()

        inference_time = time.perf_counter() - start_time

        return predictions[0], inference_time

    def filter_predictions(
        self, 
        predictions: Dict
    ) -> Tuple[List[int], List[List[float]], List[float]]:
        """
        Filter predictions by confidence threshold

        Args:
            predictions: Raw model predictions

        Returns:
            Tuple of (labels, boxes, scores)
        """
        scores = predictions['scores'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()
        boxes = predictions['boxes'].cpu().numpy()

        mask = scores >= self.confidence_threshold

        return (
            labels[mask].tolist(),
            boxes[mask].tolist(),
            scores[mask].tolist()
        )

    def draw_predictions(
        self, 
        image: Image.Image, 
        labels: List[int], 
        boxes: List[List[float]], 
        scores: List[float]
    ) -> Image.Image:
        """
        Draw bounding boxes and labels on image

        Args:
            image: Original PIL image
            labels: List of class labels
            boxes: List of bounding boxes
            scores: List of confidence scores

        Returns:
            Annotated image
        """
        image_copy = image.copy()
        draw = ImageDraw.Draw(image_copy)

        # Try to load better font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()

        # Generate colors for each category
        colors = plt.cm.hsv(np.linspace(0, 1, len(COCO_INSTANCE_CATEGORY_NAMES)))

        for label, box, score in zip(labels, boxes, scores):
            x1, y1, x2, y2 = box
            category_name = COCO_INSTANCE_CATEGORY_NAMES[label]

            # Get color for this class
            color = tuple((np.array(colors[label % len(colors)]) * 255).astype(int)[:3])

            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # Prepare label text
            label_text = f"{category_name}: {score:.2f}"

            # Get text bounding box
            try:
                text_bbox = draw.textbbox((x1, y1), label_text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
            except:
                text_width, text_height = 150, 20

            # Draw label background
            draw.rectangle(
                [x1, y1 - text_height - 4, x1 + text_width + 4, y1],
                fill=color
            )

            # Draw label text
            draw.text((x1 + 2, y1 - text_height - 2), label_text, fill='white', font=font)

        return image_copy

    def create_result_dict(
        self,
        image_name: str,
        labels: List[int],
        boxes: List[List[float]],
        scores: List[float],
        inference_time: float,
        use_gpu: bool
    ) -> Dict:
        """
        Create result dictionary for single image

        Args:
            image_name: Name of image file
            labels: List of detected class labels
            boxes: List of bounding boxes
            scores: List of confidence scores
            inference_time: Time taken for inference
            use_gpu: Whether GPU was used

        Returns:
            Result dictionary
        """
        detections = []
        for label, box, score in zip(labels, boxes, scores):
            detections.append({
                'category_name': COCO_INSTANCE_CATEGORY_NAMES[label],
                'category_id': int(label),
                'bounding_box': {
                    'x1': float(box[0]),
                    'y1': float(box[1]),
                    'x2': float(box[2]),
                    'y2': float(box[3])
                },
                'confidence_score': float(score)
            })

        return {
            'image_name': image_name,
            'detections': detections,
            'num_detections': len(detections),
            'inference_time': float(inference_time),
            'fps': float(1.0 / inference_time) if inference_time > 0 else 0.0,
            'gpu': 1 if use_gpu else 0,
            'avg_confidence': float(np.mean(scores)) if scores else 0.0,
            'max_confidence': float(np.max(scores)) if scores else 0.0,
            'min_confidence': float(np.min(scores)) if scores else 0.0
        }

    def process_images(
        self,
        input_folder: str,
        output_folder: str,
        use_gpu: bool = False
    ) -> List[Dict]:
        """
        Process all images in folder

        Args:
            input_folder: Path to input images
            output_folder: Path to save results
            use_gpu: Whether to use GPU

        Returns:
            List of result dictionaries
        """
        # Get image files
        image_files = get_image_files(input_folder)

        if not image_files:
            print(f"No images found in {input_folder}")
            return []

        print(f"Found {len(image_files)} images to process")

        # Load model
        model = self.load_model(use_gpu=use_gpu)
        device = self.device_gpu if use_gpu else self.device_cpu

        # Create output directory
        output_images_dir = Path(output_folder) / 'images_baseline'
        output_images_dir.mkdir(parents=True, exist_ok=True)

        results = []
        total_start_time = time.time()

        print()
        print(f"Processing with {'GPU' if use_gpu else 'CPU'}:")
        print("-" * 80)

        for idx, image_path in enumerate(image_files, 1):
            try:
                # Load and preprocess
                image_tensor, original_image = self.preprocess_image(str(image_path))

                # Run inference
                predictions, inference_time = self.run_inference(
                    image_tensor, model, device
                )

                # Filter predictions
                labels, boxes, scores = self.filter_predictions(predictions)

                # Print progress
                print(f"[{idx}/{len(image_files)}] {image_path.name:40s} | "
                      f"Objects: {len(labels):3d} | "
                      f"Time: {inference_time:.4f}s | "
                      f"FPS: {1.0/inference_time:6.2f}")

                # Draw and save annotated image
                annotated_image = self.draw_predictions(
                    original_image, labels, boxes, scores
                )
                output_path = output_images_dir / image_path.name
                annotated_image.save(output_path, quality=95)

                # Create result dictionary
                result = self.create_result_dict(
                    image_path.name, labels, boxes, scores, inference_time, use_gpu
                )
                results.append(result)

            except Exception as e:
                print(f"Error processing {image_path.name}: {str(e)}")
                continue

        total_time = time.time() - total_start_time

        print("-" * 80)
        print(f"Processing completed in {total_time:.2f}s")
        print()

        return results


# ============================================================================
# STATISTICS AND ANALYSIS
# ============================================================================

def compute_statistics(results: List[Dict], device_name: str) -> Dict:
    """
    Compute comprehensive statistics from results

    Args:
        results: List of result dictionaries
        device_name: Name of device (CPU/GPU)

    Returns:
        Statistics dictionary
    """
    if not results:
        return {}

    inference_times = [r['inference_time'] for r in results]
    fps_values = [r['fps'] for r in results]
    num_detections = [r['num_detections'] for r in results]
    avg_confidences = [r['avg_confidence'] for r in results if r['avg_confidence'] > 0]

    # Category distribution
    category_counts = {}
    for result in results:
        for detection in result['detections']:
            cat_name = detection['category_name']
            category_counts[cat_name] = category_counts.get(cat_name, 0) + 1

    return {
        'device': device_name,
        'total_images': len(results),
        'total_detections': sum(num_detections),
        'avg_detections_per_image': float(np.mean(num_detections)),
        'std_detections_per_image': float(np.std(num_detections)),
        'total_inference_time': float(sum(inference_times)),
        'avg_inference_time': float(np.mean(inference_times)),
        'std_inference_time': float(np.std(inference_times)),
        'min_inference_time': float(min(inference_times)),
        'max_inference_time': float(max(inference_times)),
        'median_inference_time': float(np.median(inference_times)),
        'avg_fps': float(np.mean(fps_values)),
        'std_fps': float(np.std(fps_values)),
        'avg_confidence': float(np.mean(avg_confidences)) if avg_confidences else 0.0,
        'std_confidence': float(np.std(avg_confidences)) if avg_confidences else 0.0,
        'category_distribution': category_counts
    }


def generate_statistics(results_cpu: List[Dict], results_gpu: List[Dict]) -> Dict:
    """
    Generate complete statistics for both CPU and GPU

    Args:
        results_cpu: CPU inference results
        results_gpu: GPU inference results

    Returns:
        Complete statistics dictionary
    """
    stats = {
        'cpu_stats': compute_statistics(results_cpu, 'CPU'),
        'gpu_stats': compute_statistics(results_gpu, 'GPU') if results_gpu else {},
        'speedup': 0.0
    }

    # Calculate speedup
    if results_cpu and results_gpu:
        cpu_avg = stats['cpu_stats']['avg_inference_time']
        gpu_avg = stats['gpu_stats']['avg_inference_time']
        stats['speedup'] = float(cpu_avg / gpu_avg) if gpu_avg > 0 else 0.0

    return stats


# ============================================================================
# VISUALIZATION
# ============================================================================

def generate_visualizations(
    results_cpu: List[Dict], 
    results_gpu: List[Dict], 
    output_folder: str
):
    """
    Generate comprehensive visualization plots

    Args:
        results_cpu: CPU inference results
        results_gpu: GPU inference results
        output_folder: Path to save plots
    """
    output_path = Path(output_folder)

    if not results_cpu:
        print("No results to visualize")
        return

    print("Generating visualization plots...")

    # ========================================================================
    # Plot 1: Comprehensive Inference Analysis
    # ========================================================================

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Baseline Inference Analysis', fontsize=16, fontweight='bold')

    # 1.1: Inference time per image
    ax = axes[0, 0]
    cpu_times = [r['inference_time'] for r in results_cpu]
    indices = range(len(cpu_times))

    ax.plot(indices, cpu_times, 'o-', label='CPU', linewidth=2, markersize=6, color='steelblue')

    if results_gpu:
        gpu_times = [r['inference_time'] for r in results_gpu]
        ax.plot(indices, gpu_times, 's-', label='GPU', linewidth=2, markersize=6, color='coral')

    ax.set_xlabel('Image Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Inference Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Inference Time per Image', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # 1.2: FPS distribution
    ax = axes[0, 1]
    cpu_fps = [r['fps'] for r in results_cpu]

    if results_gpu:
        gpu_fps = [r['fps'] for r in results_gpu]
        bp = ax.boxplot([cpu_fps, gpu_fps], labels=['CPU', 'GPU'], patch_artist=True)
        colors = ['lightblue', 'lightcoral']
    else:
        bp = ax.boxplot([cpu_fps], labels=['CPU'], patch_artist=True)
        colors = ['lightblue']

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_ylabel('FPS (Frames Per Second)', fontsize=12, fontweight='bold')
    ax.set_title('FPS Distribution', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 1.3: Detections per image
    ax = axes[1, 0]
    cpu_dets = [r['num_detections'] for r in results_cpu]

    ax.bar(indices, cpu_dets, alpha=0.7, color='seagreen', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Image Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Detections', fontsize=12, fontweight='bold')
    ax.set_title('Objects Detected per Image', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 1.4: Confidence distribution
    ax = axes[1, 1]
    cpu_conf = [r['avg_confidence'] for r in results_cpu if r['avg_confidence'] > 0]

    if cpu_conf:
        ax.hist(cpu_conf, bins=25, alpha=0.7, color='mediumpurple', edgecolor='black', linewidth=0.8)
        ax.axvline(np.mean(cpu_conf), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(cpu_conf):.3f}')

    ax.set_xlabel('Average Confidence Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Confidence Score Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path / 'inference_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: inference_analysis.png")

    # ========================================================================
    # Plot 2: Category Distribution
    # ========================================================================

    category_counts = {}
    for result in results_cpu:
        for detection in result['detections']:
            cat_name = detection['category_name']
            category_counts[cat_name] = category_counts.get(cat_name, 0) + 1

    if category_counts:
        sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        top_n = min(15, len(sorted_categories))
        categories = [c[0] for c in sorted_categories[:top_n]]
        counts = [c[1] for c in sorted_categories[:top_n]]

        fig, ax = plt.subplots(figsize=(12, 8))

        bars = ax.barh(range(len(categories)), counts, color='teal', edgecolor='black', linewidth=0.8)
        ax.set_yticks(range(len(categories)))
        ax.set_yticklabels(categories, fontsize=11)
        ax.set_xlabel('Detection Count', fontsize=13, fontweight='bold')
        ax.set_title(f'Top {top_n} Detected Object Categories', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        # Add count labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax.text(count, i, f'  {count}', va='center', fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path / 'category_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved: category_distribution.png")

    # ========================================================================
    # Plot 3: CPU vs GPU Comparison (if GPU available)
    # ========================================================================

    if results_cpu and results_gpu:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('CPU vs GPU Performance Comparison', fontsize=16, fontweight='bold')

        # 3.1: Average inference time
        ax = axes[0]
        cpu_avg = np.mean([r['inference_time'] for r in results_cpu])
        gpu_avg = np.mean([r['inference_time'] for r in results_gpu])

        bars = ax.bar(['CPU', 'GPU'], [cpu_avg, gpu_avg], 
                     color=['steelblue', 'coral'], edgecolor='black', linewidth=1.5, width=0.6)
        ax.set_ylabel('Average Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Average Inference Time', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        for bar, value in zip(bars, [cpu_avg, gpu_avg]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.4f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')

        # 3.2: Average FPS
        ax = axes[1]
        cpu_fps_avg = np.mean([r['fps'] for r in results_cpu])
        gpu_fps_avg = np.mean([r['fps'] for r in results_gpu])

        bars = ax.bar(['CPU', 'GPU'], [cpu_fps_avg, gpu_fps_avg], 
                     color=['steelblue', 'coral'], edgecolor='black', linewidth=1.5, width=0.6)
        ax.set_ylabel('Average FPS', fontsize=12, fontweight='bold')
        ax.set_title('Average Frames Per Second', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        for bar, value in zip(bars, [cpu_fps_avg, gpu_fps_avg]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        # 3.3: Speedup factor
        ax = axes[2]
        speedup = cpu_avg / gpu_avg if gpu_avg > 0 else 0

        ax.bar(['Speedup'], [speedup], color='green', edgecolor='black', linewidth=1.5, width=0.4)
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='No speedup')
        ax.set_ylabel('Speedup Factor (x)', fontsize=12, fontweight='bold')
        ax.set_title('GPU Speedup', fontsize=13, fontweight='bold')
        ax.set_ylim(0, max(speedup * 1.2, 2))
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        ax.text(0, speedup, f'{speedup:.2f}x', ha='center', va='bottom', 
               fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path / 'cpu_vs_gpu_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved: cpu_vs_gpu_comparison.png")

    print()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def print_summary(stats: Dict, results_cpu: List[Dict], results_gpu: List[Dict]):
    """Print summary statistics"""

    print_header("INFERENCE SUMMARY")

    if results_cpu:
        print_section("CPU Results")
        cpu_stats = stats['cpu_stats']
        print(f"Total images processed:        {cpu_stats['total_images']}")
        print(f"Total objects detected:        {cpu_stats['total_detections']}")
        print(f"Average detections per image:  {cpu_stats['avg_detections_per_image']:.2f} ± {cpu_stats['std_detections_per_image']:.2f}")
        print(f"Total inference time:          {cpu_stats['total_inference_time']:.2f}s")
        print(f"Average inference time:        {cpu_stats['avg_inference_time']:.4f}s ± {cpu_stats['std_inference_time']:.4f}s")
        print(f"Median inference time:         {cpu_stats['median_inference_time']:.4f}s")
        print(f"Min/Max inference time:        {cpu_stats['min_inference_time']:.4f}s / {cpu_stats['max_inference_time']:.4f}s")
        print(f"Average FPS:                   {cpu_stats['avg_fps']:.2f} ± {cpu_stats['std_fps']:.2f}")
        print(f"Average confidence:            {cpu_stats['avg_confidence']:.4f} ± {cpu_stats['std_confidence']:.4f}")
        print(f"Unique categories detected:    {len(cpu_stats['category_distribution'])}")

    if results_gpu:
        print_section("GPU Results")
        gpu_stats = stats['gpu_stats']
        print(f"Total images processed:        {gpu_stats['total_images']}")
        print(f"Total objects detected:        {gpu_stats['total_detections']}")
        print(f"Average detections per image:  {gpu_stats['avg_detections_per_image']:.2f} ± {gpu_stats['std_detections_per_image']:.2f}")
        print(f"Total inference time:          {gpu_stats['total_inference_time']:.2f}s")
        print(f"Average inference time:        {gpu_stats['avg_inference_time']:.4f}s ± {gpu_stats['std_inference_time']:.4f}s")
        print(f"Median inference time:         {gpu_stats['median_inference_time']:.4f}s")
        print(f"Min/Max inference time:        {gpu_stats['min_inference_time']:.4f}s / {gpu_stats['max_inference_time']:.4f}s")
        print(f"Average FPS:                   {gpu_stats['avg_fps']:.2f} ± {gpu_stats['std_fps']:.2f}")
        print(f"Average confidence:            {gpu_stats['avg_confidence']:.4f} ± {gpu_stats['std_confidence']:.4f}")
        print(f"Unique categories detected:    {len(gpu_stats['category_distribution'])}")

        print_section("Performance Gain")
        print(f"GPU Speedup:                   {stats['speedup']:.2f}x")
        time_saved = cpu_stats['total_inference_time'] - gpu_stats['total_inference_time']
        time_saved_pct = (time_saved / cpu_stats['total_inference_time']) * 100
        print(f"Time saved:                    {time_saved:.2f}s ({time_saved_pct:.1f}%)")


def main():
    """
    Main execution function
    """
    # Configuration
    INPUT_FOLDER = 'subset_validation/images'
    OUTPUT_FOLDER = 'subset_validation/baseline_result'
    CONFIDENCE_THRESHOLD = 0.5

    # Print header
    print_header("SSDLite320 MobileNetV3 Large - BASELINE INFERENCE", "=", 80)

    print(f"Configuration:")
    print(f"  Input folder:          {INPUT_FOLDER}")
    print(f"  Output folder:         {OUTPUT_FOLDER}")
    print(f"  Confidence threshold:  {CONFIDENCE_THRESHOLD}")

    # Check environment
    check_environment()

    # Verify input folder exists
    if not Path(INPUT_FOLDER).exists():
        print()
        print(f"ERROR: Input folder '{INPUT_FOLDER}' does not exist!")
        print(f"Please create the folder and add validation images.")
        print()
        print(f"Example:")
        print(f"  mkdir -p {INPUT_FOLDER}")
        print(f"  cp your_images/*.jpg {INPUT_FOLDER}/")
        return

    # Create output directories
    create_output_directories(OUTPUT_FOLDER)

    # Initialize inference engine
    print()
    inference_engine = BaselineInferenceEngine(confidence_threshold=CONFIDENCE_THRESHOLD)

    # Process with CPU
    print_header("CPU INFERENCE", "=", 80)
    results_cpu = inference_engine.process_images(
        INPUT_FOLDER, OUTPUT_FOLDER, use_gpu=False
    )

    # Process with GPU (if available)
    results_gpu = []
    if torch.cuda.is_available():
        print_header("GPU INFERENCE", "=", 80)
        results_gpu = inference_engine.process_images(
            INPUT_FOLDER, OUTPUT_FOLDER, use_gpu=True
        )

    # Generate statistics
    print_header("GENERATING STATISTICS AND VISUALIZATIONS")
    stats = generate_statistics(results_cpu, results_gpu)

    # Save results to JSON
    output_data = {
        'model_name': 'ssdlite320_mobilenet_v3_large',
        'model_weights': 'COCO_V1',
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'num_classes': 91,
        'input_folder': INPUT_FOLDER,
        'output_folder': OUTPUT_FOLDER,
        'results_cpu': results_cpu,
        'results_gpu': results_gpu,
        'statistics': stats,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    json_path = Path(OUTPUT_FOLDER) / 'baseline_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {json_path}")
    print()

    # Generate visualizations
    generate_visualizations(results_cpu, results_gpu, OUTPUT_FOLDER)

    # Print summary
    print_summary(stats, results_cpu, results_gpu)

    # Final message
    print()
    print_header("BASELINE INFERENCE COMPLETED SUCCESSFULLY", "=", 80)
    print()
    print("Output files:")
    print(f"  - JSON results:      {OUTPUT_FOLDER}/baseline_results.json")
    print(f"  - Annotated images:  {OUTPUT_FOLDER}/images_baseline/")
    print(f"  - Analysis plots:    {OUTPUT_FOLDER}/*.png")
    print()


if __name__ == '__main__':
    main()
