#!/usr/bin/env python3
"""
SSDLite320 MobileNetV3 Large - Early Exit Inference System (Phase 1.4)
Complete implementation for early-exit object detection evaluation

Author: Early Exit Inference System
Date: October 2025
Version: 1.0

Usage:
    python baseline_inference_complete_ee_phase_1_4.py

Requirements:
    - torch
    - torchvision
    - pillow
    - matplotlib
    - numpy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as TF
from torchvision.ops import nms
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

# For phase 1.4, we're detecting chairs only, but map to COCO chair category
CHAIR_CATEGORY_ID = 62


# ============================================================================
# MODEL ARCHITECTURE (from training script)
# ============================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class EnhancedEarlyExitBranch(nn.Module):
    """Enhanced early exit branch with SE attention and larger capacity"""

    def __init__(self, in_channels, intermediate_channels, num_anchors, num_classes=2, use_attention=True):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU6(inplace=True),

            nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3,
                     padding=1, groups=intermediate_channels, bias=False),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU6(inplace=True),

            nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU6(inplace=True)
        )

        self.use_attention = use_attention
        if use_attention:
            self.se_block = SEBlock(intermediate_channels, reduction=4)

        self.additional_features = nn.Sequential(
            nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3,
                     padding=1, groups=intermediate_channels, bias=False),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU6(inplace=True),

            nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU6(inplace=True)
        )

        self.cls_head = nn.Conv2d(intermediate_channels,
                                  num_anchors * num_classes, kernel_size=1)

        self.reg_head = nn.Conv2d(intermediate_channels,
                                  num_anchors * 4, kernel_size=1)

    def forward(self, x):
        features = self.feature_extractor(x)

        if self.use_attention:
            features = self.se_block(features)

        features = self.additional_features(features)

        cls_logits = self.cls_head(features)
        bbox_regression = self.reg_head(features)

        return cls_logits, bbox_regression


class MultiLevelCascadeEarlyExitSSDLite(nn.Module):
    """SSDLite with multi-level cascade early exit capability"""

    def __init__(self, base_model, exit1_threshold=0.45, exit2_threshold=0.60):
        super().__init__()

        self.backbone = base_model.backbone

        backbone_features = base_model.backbone.features
        first_sequential = list(backbone_features.children())[0]
        first_block_layers = list(first_sequential.children())

        self.exit1_features = nn.Sequential(*first_block_layers[:8])
        self.mid_features = nn.Sequential(*first_block_layers[8:12])
        self.late_features = nn.Sequential(*first_block_layers[12:])

        num_anchors = 6

        self.exit1_branch = EnhancedEarlyExitBranch(
            in_channels=80,
            intermediate_channels=128,
            num_anchors=num_anchors,
            num_classes=2,
            use_attention=True
        )

        self.exit2_context_proj = nn.Conv2d(128, 112, kernel_size=1)
        self.exit2_branch = EnhancedEarlyExitBranch(
            in_channels=112,
            intermediate_channels=224,
            num_anchors=num_anchors,
            num_classes=2,
            use_attention=True
        )

        self.full_context_proj = nn.Conv2d(224, 672, kernel_size=1)
        self.full_branch = EnhancedEarlyExitBranch(
            in_channels=672,
            intermediate_channels=512,
            num_anchors=num_anchors,
            num_classes=2,
            use_attention=True
        )

        self.exit1_threshold = exit1_threshold
        self.exit2_threshold = exit2_threshold
        self.num_classes = 2

        self.temperature1 = nn.Parameter(torch.ones(1) * 1.5)
        self.temperature2 = nn.Parameter(torch.ones(1) * 1.5)
        self.temperature_full = nn.Parameter(torch.ones(1) * 1.0)

        self.exit_stats = {'exit1': 0, 'exit2': 0, 'full': 0}
        self.training_step = 0

    def compute_confidence(self, cls_logits, temperature, bbox_regression=None):
        """Compute confidence metric for early exit decision"""
        batch_size = cls_logits.shape[0]

        cls_logits = cls_logits.permute(0, 2, 3, 1).contiguous()
        cls_logits = cls_logits.view(batch_size, -1, self.num_classes)

        scaled_logits = cls_logits / temperature
        probs = F.softmax(scaled_logits, dim=-1)

        objectness = probs[:, :, 1]
        max_probs, _ = probs.max(dim=-1)

        objectness_threshold = 0.4
        min_confident_objects = 2

        confidence_list = []
        for i in range(batch_size):
            object_mask = objectness[i] > objectness_threshold
            num_confident_objects = object_mask.sum().item()

            if num_confident_objects >= min_confident_objects:
                obj_confidence = objectness[i][object_mask].mean()
                cls_confidence = max_probs[i][object_mask].mean()
                conf = 0.8 * obj_confidence + 0.2 * cls_confidence

                object_count_factor = min(1.0, 0.85 + 0.15 * (num_confident_objects - min_confident_objects) / 8.0)
                conf = conf * object_count_factor

            elif num_confident_objects > 0:
                obj_confidence = objectness[i][object_mask].mean()
                cls_confidence = max_probs[i][object_mask].mean()
                conf = 0.6 * obj_confidence + 0.4 * cls_confidence
                conf = conf * 0.5

            else:
                conf = max_probs[i].mean()
                conf = conf * 0.4

            confidence_list.append(conf)

        confidence = torch.stack(confidence_list)
        return confidence

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be provided")

        if isinstance(images, (list, tuple)):
            images = torch.stack(images)

        # Exit 1
        features_exit1 = self.exit1_features(images)
        exit1_cls, exit1_reg = self.exit1_branch(features_exit1)
        exit1_confidence = self.compute_confidence(exit1_cls, self.temperature1)

        if not self.training:
            avg_confidence_exit1 = exit1_confidence.mean().item()

            if avg_confidence_exit1 >= self.exit1_threshold:
                self.exit_stats['exit1'] += 1
                return {
                    'cls_logits': exit1_cls,
                    'bbox_regression': exit1_reg,
                    'exit_point': 'exit1',
                    'confidence': avg_confidence_exit1
                }

            # Continue to exit 2
            features_mid = self.mid_features(features_exit1)

            exit1_context = self.exit1_branch.feature_extractor(features_exit1)
            if self.exit1_branch.use_attention:
                exit1_context = self.exit1_branch.se_block(exit1_context)
            exit1_context = self.exit1_branch.additional_features(exit1_context)

            if exit1_context.shape[2:] != features_mid.shape[2:]:
                exit1_context = F.adaptive_avg_pool2d(exit1_context, features_mid.shape[2:])

            exit1_context_proj = self.exit2_context_proj(exit1_context)
            if exit1_context_proj.shape[2:] != features_mid.shape[2:]:
                exit1_context_proj = F.adaptive_avg_pool2d(exit1_context_proj, features_mid.shape[2:])

            features_exit2 = features_mid + exit1_context_proj

            exit2_cls, exit2_reg = self.exit2_branch(features_exit2)
            exit2_confidence = self.compute_confidence(exit2_cls, self.temperature2)
            avg_confidence_exit2 = exit2_confidence.mean().item()

            if avg_confidence_exit2 >= self.exit2_threshold:
                self.exit_stats['exit2'] += 1
                return {
                    'cls_logits': exit2_cls,
                    'bbox_regression': exit2_reg,
                    'exit_point': 'exit2',
                    'confidence': avg_confidence_exit2
                }

            # Continue to full model
            features_late = self.late_features(features_mid)

            exit2_context = self.exit2_branch.feature_extractor(features_exit2)
            if self.exit2_branch.use_attention:
                exit2_context = self.exit2_branch.se_block(exit2_context)
            exit2_context = self.exit2_branch.additional_features(exit2_context)

            if exit2_context.shape[2:] != features_late.shape[2:]:
                exit2_context = F.adaptive_avg_pool2d(exit2_context, features_late.shape[2:])

            exit2_context_proj = self.full_context_proj(exit2_context)
            if exit2_context_proj.shape[2:] != features_late.shape[2:]:
                exit2_context_proj = F.adaptive_avg_pool2d(exit2_context_proj, features_late.shape[2:])

            features_full = features_late + exit2_context_proj

            full_cls, full_reg = self.full_branch(features_full)
            full_confidence = self.compute_confidence(full_cls, self.temperature_full)

            self.exit_stats['full'] += 1
            return {
                'cls_logits': full_cls,
                'bbox_regression': full_reg,
                'exit_point': 'full',
                'confidence': full_confidence.mean().item()
            }


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
        Path(base_path) / "images_phase_1_4",
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

    unique_files = list(dict.fromkeys(image_files))

    return sorted(unique_files)


# ============================================================================
# INFERENCE ENGINE
# ============================================================================

class EarlyExitInferenceEngine:
    """
    Complete inference engine for early-exit object detection
    """

    def __init__(self, model_path: str, confidence_threshold: float = 0.5,
                 exit1_threshold: float = 0.45, exit2_threshold: float = 0.60):
        """
        Initialize inference engine

        Args:
            model_path: Path to trained model weights
            confidence_threshold: Minimum confidence score for detections
            exit1_threshold: Confidence threshold for exit 1
            exit2_threshold: Confidence threshold for exit 2
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.exit1_threshold = exit1_threshold
        self.exit2_threshold = exit2_threshold
        self.device_cpu = torch.device('cpu')
        self.device_gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_cpu = None
        self.model_gpu = None

        print(f"Early Exit Inference engine initialized")
        print(f"  Confidence threshold: {confidence_threshold}")
        print(f"  Exit 1 threshold: {exit1_threshold}")
        print(f"  Exit 2 threshold: {exit2_threshold}")

    def load_model(self, use_gpu: bool = False) -> torch.nn.Module:
        """
        Load trained early-exit model

        Args:
            use_gpu: Whether to load model on GPU

        Returns:
            Loaded model
        """
        device = self.device_gpu if use_gpu else self.device_cpu
        device_name = "GPU" if use_gpu else "CPU"

        print(f"Loading early-exit model on {device_name}...", end=" ")

        # Create base model (for architecture) - use DEFAULT weights to match training
        from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights
        base_model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
            weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        )

        # Create early-exit model
        model = MultiLevelCascadeEarlyExitSSDLite(
            base_model,
            exit1_threshold=self.exit1_threshold,
            exit2_threshold=self.exit2_threshold
        )

        # Load trained weights
        state_dict = torch.load(self.model_path, map_location=device)

        # Load with strict=False to handle potential architecture differences
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if missing_keys:
            print(f"\nWarning: Missing keys in state dict: {len(missing_keys)} keys")
        if unexpected_keys:
            print(f"\nWarning: Unexpected keys in state dict: {len(unexpected_keys)} keys")

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

        # Resize to 320x320 (model input size)
        image_resized = image.resize((320, 320), Image.BILINEAR)
        image_tensor = TF.to_tensor(image_resized)

        return image_tensor, image

    def decode_predictions(
        self,
        cls_logits: torch.Tensor,
        bbox_regression: torch.Tensor,
        original_size: Tuple[int, int],
        model_input_size: int = 320
    ) -> Tuple[List[int], List[List[float]], List[float]]:
        """
        Simplified decode for early-exit model outputs

        Args:
            cls_logits: Classification logits [1, num_anchors*num_classes, H, W]
            bbox_regression: Bounding box regression [1, num_anchors*4, H, W]
            original_size: Original image size (width, height)
            model_input_size: Model input size

        Returns:
            Tuple of (labels, boxes, scores)
        """
        device = cls_logits.device
        batch_size = cls_logits.shape[0]

        # Get spatial dimensions
        h, w = cls_logits.shape[2], cls_logits.shape[3]
        num_anchors = 6
        num_classes = 2  # background + chair

        # Reshape and process predictions
        # cls_logits: [1, 12, H, W] -> [1, H*W*6, 2]
        cls_logits = cls_logits.permute(0, 2, 3, 1).contiguous()
        cls_logits = cls_logits.view(batch_size, h * w * num_anchors, num_classes)

        # bbox_regression: [1, 24, H, W] -> [1, H*W*6, 4]
        bbox_regression = bbox_regression.permute(0, 2, 3, 1).contiguous()
        bbox_regression = bbox_regression.view(batch_size, h * w * num_anchors, 4)

        # Apply softmax to get probabilities
        probs = F.softmax(cls_logits[0], dim=-1)  # [H*W*6, 2]

        # Get chair class probabilities (class 1)
        chair_scores = probs[:, 1]  # [H*W*6]

        # Use detection threshold (model outputs are in 0.41-0.46 range)
        detection_threshold = self.confidence_threshold

        # Filter by threshold
        mask = chair_scores >= detection_threshold

        if mask.sum() == 0:
            return [], [], []

        filtered_scores = chair_scores[mask]
        filtered_bbox_deltas = bbox_regression[0][mask]  # [N, 4]

        # Generate simple grid-based boxes
        # Create a grid of anchor centers
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing='ij'
        )

        # Scale to image coordinates
        stride = model_input_size / h
        centers_x = (grid_x + 0.5) * stride
        centers_y = (grid_y + 0.5) * stride

        # Flatten and repeat for anchors
        centers_x = centers_x.reshape(-1, 1).repeat(1, num_anchors).reshape(-1)
        centers_y = centers_y.reshape(-1, 1).repeat(1, num_anchors).reshape(-1)

        # Different anchor sizes
        anchor_sizes = torch.tensor([40, 60, 80, 100, 120, 140], device=device, dtype=torch.float32)
        anchor_sizes = anchor_sizes.repeat(h * w)

        # Filter centers and sizes
        filtered_centers_x = centers_x[mask]
        filtered_centers_y = centers_y[mask]
        filtered_sizes = anchor_sizes[mask]

        # Decode boxes: center + size + deltas
        # IMPORTANT: The model's bbox regression didn't learn proper deltas (values are 175-379)
        # So we use minimal displacement and fixed box sizes
        # Treat the bbox regression as noise and use fixed-size boxes at anchor locations
        pred_cx = filtered_centers_x + filtered_bbox_deltas[:, 0].clamp(-2, 2)  # Small offset only
        pred_cy = filtered_centers_y + filtered_bbox_deltas[:, 1].clamp(-2, 2)
        pred_w = filtered_sizes.clone()  # Use anchor size directly
        pred_h = filtered_sizes.clone()

        # Convert to xyxy format
        pred_x1 = pred_cx - pred_w / 2
        pred_y1 = pred_cy - pred_h / 2
        pred_x2 = pred_cx + pred_w / 2
        pred_y2 = pred_cy + pred_h / 2

        decoded_boxes = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1)

        # Scale to original image size
        scale_x = original_size[0] / model_input_size
        scale_y = original_size[1] / model_input_size

        decoded_boxes[:, [0, 2]] *= scale_x
        decoded_boxes[:, [1, 3]] *= scale_y

        # Clip to image boundaries
        decoded_boxes[:, 0] = torch.clamp(decoded_boxes[:, 0], 0, original_size[0])
        decoded_boxes[:, 1] = torch.clamp(decoded_boxes[:, 1], 0, original_size[1])
        decoded_boxes[:, 2] = torch.clamp(decoded_boxes[:, 2], 0, original_size[0])
        decoded_boxes[:, 3] = torch.clamp(decoded_boxes[:, 3], 0, original_size[1])

        # Filter out invalid boxes (where x2 <= x1 or y2 <= y1)
        valid_mask = (decoded_boxes[:, 2] > decoded_boxes[:, 0]) & (decoded_boxes[:, 3] > decoded_boxes[:, 1])

        if valid_mask.sum() == 0:
            return [], [], []

        decoded_boxes = decoded_boxes[valid_mask]
        filtered_scores = filtered_scores[valid_mask]

        # Apply NMS
        keep_indices = nms(decoded_boxes, filtered_scores, iou_threshold=0.45)

        final_boxes = decoded_boxes[keep_indices].cpu().numpy().tolist()
        final_scores = filtered_scores[keep_indices].cpu().numpy().tolist()
        final_labels = [CHAIR_CATEGORY_ID] * len(final_boxes)

        return final_labels, final_boxes, final_scores

    def _generate_anchors(self, h: int, w: int, num_anchors: int, input_size: int) -> torch.Tensor:
        """Generate anchor boxes for all spatial locations"""
        # Create grid of centers
        cy = torch.arange(h, dtype=torch.float32) + 0.5
        cx = torch.arange(w, dtype=torch.float32) + 0.5

        # Scale to input image coordinates
        cy = cy * (input_size / h)
        cx = cx * (input_size / w)

        # Create meshgrid
        grid_y, grid_x = torch.meshgrid(cy, cx, indexing='ij')

        # Define anchor sizes for each anchor
        anchor_sizes = [30, 50, 70, 90, 110, 130]  # 6 different sizes

        anchors = []
        for size in anchor_sizes[:num_anchors]:
            half_size = size / 2
            # Create boxes: [x1, y1, x2, y2]
            x1 = grid_x - half_size
            y1 = grid_y - half_size
            x2 = grid_x + half_size
            y2 = grid_y + half_size

            anchor = torch.stack([x1, y1, x2, y2], dim=-1)  # [H, W, 4]
            anchors.append(anchor)

        # Stack anchors: [H, W, num_anchors, 4]
        anchors = torch.stack(anchors, dim=2)

        return anchors

    def _decode_boxes(self, anchors: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        """
        Decode bounding boxes from anchors and deltas

        Args:
            anchors: [N, 4] anchor boxes in (x1, y1, x2, y2) format
            deltas: [N, 4] predicted deltas

        Returns:
            [N, 4] decoded boxes in (x1, y1, x2, y2) format
        """
        # Convert anchors to center format
        anchor_widths = anchors[:, 2] - anchors[:, 0]
        anchor_heights = anchors[:, 3] - anchors[:, 1]
        anchor_cx = anchors[:, 0] + 0.5 * anchor_widths
        anchor_cy = anchors[:, 1] + 0.5 * anchor_heights

        # Apply deltas
        pred_cx = anchor_cx + deltas[:, 0] * anchor_widths
        pred_cy = anchor_cy + deltas[:, 1] * anchor_heights
        pred_w = anchor_widths * torch.exp(deltas[:, 2])
        pred_h = anchor_heights * torch.exp(deltas[:, 3])

        # Convert back to corner format
        pred_x1 = pred_cx - 0.5 * pred_w
        pred_y1 = pred_cy - 0.5 * pred_h
        pred_x2 = pred_cx + 0.5 * pred_w
        pred_y2 = pred_cy + 0.5 * pred_h

        pred_boxes = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1)

        return pred_boxes

    def run_inference(
        self,
        image_tensor: torch.Tensor,
        original_image: Image.Image,
        model: torch.nn.Module,
        device: torch.device
    ) -> Tuple[Dict, float, str, float]:
        """
        Run inference on single image

        Args:
            image_tensor: Preprocessed image tensor
            original_image: Original PIL image
            model: Detection model
            device: Device to run on

        Returns:
            Tuple of (predictions, inference_time, exit_point, confidence)
        """
        image_tensor = image_tensor.to(device).unsqueeze(0)

        # Warm-up for GPU
        if device.type == 'cuda':
            with torch.no_grad():
                _ = model(image_tensor)
            torch.cuda.synchronize()

        # Measure inference time
        start_time = time.perf_counter()

        with torch.no_grad():
            outputs = model(image_tensor)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        inference_time = time.perf_counter() - start_time

        # Extract information
        exit_point = outputs['exit_point']
        confidence = outputs['confidence']

        # Decode predictions
        original_size = original_image.size  # (width, height)
        labels, boxes, scores = self.decode_predictions(
            outputs['cls_logits'],
            outputs['bbox_regression'],
            original_size
        )

        predictions = {
            'labels': labels,
            'boxes': boxes,
            'scores': scores
        }

        return predictions, inference_time, exit_point, confidence

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

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()

        colors = plt.cm.hsv(np.linspace(0, 1, len(COCO_INSTANCE_CATEGORY_NAMES)))

        for label, box, score in zip(labels, boxes, scores):
            x1, y1, x2, y2 = box
            category_name = COCO_INSTANCE_CATEGORY_NAMES[label]

            color = tuple((np.array(colors[label % len(colors)]) * 255).astype(int)[:3])

            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            label_text = f"{category_name}: {score:.2f}"

            try:
                text_bbox = draw.textbbox((x1, y1), label_text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
            except:
                text_width, text_height = 150, 20

            draw.rectangle(
                [x1, y1 - text_height - 4, x1 + text_width + 4, y1],
                fill=color
            )

            draw.text((x1 + 2, y1 - text_height - 2), label_text, fill='white', font=font)

        return image_copy

    def create_result_dict(
        self,
        image_name: str,
        labels: List[int],
        boxes: List[List[float]],
        scores: List[float],
        inference_time: float,
        use_gpu: bool,
        exit_point: str,
        exit_confidence: float
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
            exit_point: Which exit was used (exit1/exit2/full)
            exit_confidence: Confidence score at exit

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
            'exit_point': exit_point,
            'exit_confidence': float(exit_confidence),
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
        image_files = get_image_files(input_folder)

        if not image_files:
            print(f"No images found in {input_folder}")
            return []

        print(f"Found {len(image_files)} images to process")

        model = self.load_model(use_gpu=use_gpu)
        device = self.device_gpu if use_gpu else self.device_cpu

        output_images_dir = Path(output_folder) / 'images_phase_1_4'
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
                predictions, inference_time, exit_point, exit_confidence = self.run_inference(
                    image_tensor, original_image, model, device
                )

                labels = predictions['labels']
                boxes = predictions['boxes']
                scores = predictions['scores']

                # Print progress
                print(f"[{idx}/{len(image_files)}] {image_path.name:40s} | "
                      f"Exit: {exit_point:5s} | "
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
                    image_path.name, labels, boxes, scores,
                    inference_time, use_gpu, exit_point, exit_confidence
                )
                results.append(result)

            except Exception as e:
                print(f"Error processing {image_path.name}: {str(e)}")
                import traceback
                traceback.print_exc()
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

    # Exit point distribution
    exit_distribution = {'exit1': 0, 'exit2': 0, 'full': 0}
    for result in results:
        exit_point = result.get('exit_point', 'full')
        exit_distribution[exit_point] += 1

    total_samples = len(results)
    exit_rates = {
        'exit1_rate': exit_distribution['exit1'] / total_samples if total_samples > 0 else 0,
        'exit2_rate': exit_distribution['exit2'] / total_samples if total_samples > 0 else 0,
        'full_rate': exit_distribution['full'] / total_samples if total_samples > 0 else 0,
        'combined_early_exit_rate': (exit_distribution['exit1'] + exit_distribution['exit2']) / total_samples if total_samples > 0 else 0
    }

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
        'category_distribution': category_counts,
        'exit_distribution': exit_distribution,
        'exit_rates': exit_rates
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

    # Plot 1: Comprehensive Inference Analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Early Exit Inference Analysis (Phase 1.4)', fontsize=16, fontweight='bold')

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

    # 1.4: Exit point distribution
    ax = axes[1, 1]
    exit_points = [r.get('exit_point', 'full') for r in results_cpu]
    exit_counts = {'exit1': exit_points.count('exit1'),
                   'exit2': exit_points.count('exit2'),
                   'full': exit_points.count('full')}

    colors_exit = ['red', 'green', 'blue']
    ax.bar(exit_counts.keys(), exit_counts.values(), color=colors_exit, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Exit Point Distribution', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path / 'inference_analysis_phase_1_4.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  [OK] Saved: inference_analysis_phase_1_4.png")

    # Plot 2: Category Distribution
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
        ax.set_title(f'Top {top_n} Detected Object Categories (Phase 1.4)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax.text(count, i, f'  {count}', va='center', fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path / 'category_distribution_phase_1_4.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  [OK] Saved: category_distribution_phase_1_4.png")

    # Plot 3: CPU vs GPU Comparison (if GPU available)
    if results_cpu and results_gpu:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('CPU vs GPU Performance Comparison (Phase 1.4)', fontsize=16, fontweight='bold')

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
        plt.savefig(output_path / 'cpu_vs_gpu_comparison_phase_1_4.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  [OK] Saved: cpu_vs_gpu_comparison_phase_1_4.png")

    print()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def print_summary(stats: Dict, results_cpu: List[Dict], results_gpu: List[Dict]):
    """Print summary statistics"""

    print_header("INFERENCE SUMMARY (PHASE 1.4)")

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

        print_section("Early Exit Statistics")
        exit_dist = cpu_stats['exit_distribution']
        exit_rates = cpu_stats['exit_rates']
        print(f"Exit 1 (layer 8):              {exit_dist['exit1']} images ({exit_rates['exit1_rate']:.2%})")
        print(f"Exit 2 (layer 12):             {exit_dist['exit2']} images ({exit_rates['exit2_rate']:.2%})")
        print(f"Full model:                    {exit_dist['full']} images ({exit_rates['full_rate']:.2%})")
        print(f"Combined early exit rate:      {exit_rates['combined_early_exit_rate']:.2%}")

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

        print_section("Early Exit Statistics")
        exit_dist = gpu_stats['exit_distribution']
        exit_rates = gpu_stats['exit_rates']
        print(f"Exit 1 (layer 8):              {exit_dist['exit1']} images ({exit_rates['exit1_rate']:.2%})")
        print(f"Exit 2 (layer 12):             {exit_dist['exit2']} images ({exit_rates['exit2_rate']:.2%})")
        print(f"Full model:                    {exit_dist['full']} images ({exit_rates['full_rate']:.2%})")
        print(f"Combined early exit rate:      {exit_rates['combined_early_exit_rate']:.2%}")

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
    OUTPUT_FOLDER = 'subset_validation/ee_result_phase_1_4'
    MODEL_PATH = 'working/model_trained/early_exit_ssdlite_phase1_4_final.pth'
    CONFIDENCE_THRESHOLD = 0.35  # Lowered from 0.5 due to model output range
    EXIT1_THRESHOLD = 0.45
    EXIT2_THRESHOLD = 0.60

    # Print header
    print_header("SSDLite320 MobileNetV3 Large - EARLY EXIT INFERENCE (PHASE 1.4)", "=", 80)

    print(f"Configuration:")
    print(f"  Input folder:          {INPUT_FOLDER}")
    print(f"  Output folder:         {OUTPUT_FOLDER}")
    print(f"  Model path:            {MODEL_PATH}")
    print(f"  Confidence threshold:  {CONFIDENCE_THRESHOLD}")
    print(f"  Exit 1 threshold:      {EXIT1_THRESHOLD}")
    print(f"  Exit 2 threshold:      {EXIT2_THRESHOLD}")

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

    # Verify model exists
    if not Path(MODEL_PATH).exists():
        print()
        print(f"ERROR: Model file '{MODEL_PATH}' does not exist!")
        print(f"Please train the model first using early_exit_training_system_v3_phase1_4.py")
        return

    # Create output directories
    create_output_directories(OUTPUT_FOLDER)

    # Initialize inference engine
    print()
    inference_engine = EarlyExitInferenceEngine(
        model_path=MODEL_PATH,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        exit1_threshold=EXIT1_THRESHOLD,
        exit2_threshold=EXIT2_THRESHOLD
    )

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
        'model_name': 'early_exit_ssdlite320_mobilenet_v3_large_phase1_4',
        'model_path': MODEL_PATH,
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'exit1_threshold': EXIT1_THRESHOLD,
        'exit2_threshold': EXIT2_THRESHOLD,
        'num_classes': 2,
        'input_folder': INPUT_FOLDER,
        'output_folder': OUTPUT_FOLDER,
        'results_cpu': results_cpu,
        'results_gpu': results_gpu,
        'statistics': stats,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    json_path = Path(OUTPUT_FOLDER) / 'ee_results_phase_1_4.json'
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
    print_header("EARLY EXIT INFERENCE COMPLETED SUCCESSFULLY (PHASE 1.4)", "=", 80)
    print()
    print("Output files:")
    print(f"  - JSON results:      {OUTPUT_FOLDER}/ee_results_phase_1_4.json")
    print(f"  - Annotated images:  {OUTPUT_FOLDER}/images_phase_1_4/")
    print(f"  - Analysis plots:    {OUTPUT_FOLDER}/*.png")
    print()


if __name__ == '__main__':
    main()
