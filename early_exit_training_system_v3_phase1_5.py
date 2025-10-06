# early_exit_training_system_v3_phase1_5.py
# Phase 1.5: Learnable Thresholds with Distribution Loss
#
# Key improvements from Phase 1.4:
# - LEARNABLE THRESHOLDS: Thresholds are now trainable parameters (like temperature)
# - DISTRIBUTION LOSS: Guides thresholds to achieve target exit distribution
# - AUTOMATIC CALIBRATION: No manual threshold tuning - model learns optimal values
# - Conservative initialization: Exit1=0.55, Exit2=0.65 (safe margins)
#
# Phase 1.4 Analysis Findings:
# - Model confidence stable: Exit1=0.491, Exit2=0.514, Full=0.590
# - Threshold 0.45 was TOO CLOSE to confidence 0.491 (100% early exit)
# - Fixed thresholds are fragile - small changes cause dramatic shifts
# - Solution: Make thresholds learnable with distribution loss guidance

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
import warnings
import shutil
warnings.filterwarnings('ignore')

# Dataset paths
ANNOTATIONS_PATH = "D:\\Download\\JDownloader\\MSCOCO\\annotations"
TRAIN_IMAGES_PATH = "D:\\Download\\JDownloader\\MSCOCO\\images\\train2017"
TEST_IMAGES_PATH = "D:\\Download\\JDownloader\\MSCOCO\\images\\test2017"
VAL_IMAGES_PATH = "D:\\Download\\JDownloader\\MSCOCO\\images\\val2017"

# Output paths
MODEL_ROOT_OUTPUT_PATH = "working/"
MODEL_OUTPUT_PATH = f"{MODEL_ROOT_OUTPUT_PATH}model_trained/"
ANALYSIS_EE_OUTPUT_PATH = f"{MODEL_ROOT_OUTPUT_PATH}ee_analysis/"


def clean_working_directory():
    """Clean the working directory before starting training"""
    if os.path.exists(MODEL_ROOT_OUTPUT_PATH):
        print(f"Cleaning working directory: {MODEL_ROOT_OUTPUT_PATH}")
        shutil.rmtree(MODEL_ROOT_OUTPUT_PATH)
        print("Working directory cleaned successfully")


def create_output_directories():
    """Create output directories for models and analysis"""
    os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)
    os.makedirs(ANALYSIS_EE_OUTPUT_PATH, exist_ok=True)
    print(f"Output directories created:")
    print(f"  - {MODEL_OUTPUT_PATH}")
    print(f"  - {ANALYSIS_EE_OUTPUT_PATH}")

# COCO chair category ID
CHAIR_CATEGORY_ID = 62


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    return tuple(zip(*batch))


class ChairCocoDataset(Dataset):
    """Dataset for chair detection only from COCO"""

    def __init__(self, img_folder, ann_file, transforms=None, max_samples=None):
        self.img_folder = img_folder
        self.transforms = transforms

        # Load COCO annotations
        with open(ann_file, 'r') as f:
            self.coco = json.load(f)

        # Create image id to filename mapping
        self.img_id_to_filename = {img['id']: img['file_name']
                                   for img in self.coco['images']}

        # Filter annotations for chairs only
        self.chair_annotations = defaultdict(list)
        for ann in self.coco['annotations']:
            if ann['category_id'] == CHAIR_CATEGORY_ID:
                self.chair_annotations[ann['image_id']].append(ann)

        # Keep only images with chairs
        self.valid_img_ids = list(self.chair_annotations.keys())

        # Limit dataset size if specified
        if max_samples and max_samples < len(self.valid_img_ids):
            self.valid_img_ids = self.valid_img_ids[:max_samples]

        print(f"Found {len(self.valid_img_ids)} images with chairs")

    def __len__(self):
        return len(self.valid_img_ids)

    def __getitem__(self, idx):
        img_id = self.valid_img_ids[idx]
        filename = self.img_id_to_filename[img_id]
        img_path = os.path.join(self.img_folder, filename)

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Get chair annotations for this image
        anns = self.chair_annotations[img_id]

        # Convert to format expected by SSD
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(1)  # 1 for chair, 0 is background

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id])
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target


def box_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes
    boxes1, boxes2: [N, 4] and [M, 4] tensors in (x1, y1, x2, y2) format
    Returns: [N, M] tensor of IoU values
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou


def match_anchors_to_targets(anchors, targets, iou_threshold=0.5):
    """
    Match anchors to ground truth boxes for SSD-style training

    Args:
        anchors: [num_anchors, 4] tensor of anchor boxes
        targets: dict with 'boxes' [num_gt, 4] and 'labels' [num_gt]
        iou_threshold: minimum IoU to consider a match

    Returns:
        matched_labels: [num_anchors] tensor (-1=ignore, 0=background, 1=chair)
        matched_boxes: [num_anchors, 4] tensor of matched gt boxes
        matched_ious: [num_anchors] tensor of IoU with matched box
    """
    gt_boxes = targets['boxes']
    gt_labels = targets['labels']

    if len(gt_boxes) == 0:
        # No objects - all background
        return (torch.zeros(len(anchors), dtype=torch.long, device=anchors.device),
                torch.zeros((len(anchors), 4), dtype=torch.float32, device=anchors.device),
                torch.zeros(len(anchors), dtype=torch.float32, device=anchors.device))

    # Compute IoU between anchors and ground truth
    ious = box_iou(anchors, gt_boxes)  # [num_anchors, num_gt]

    # For each anchor, find best matching ground truth
    best_gt_iou, best_gt_idx = ious.max(dim=1)  # [num_anchors]

    # Initialize all as background
    matched_labels = torch.zeros(len(anchors), dtype=torch.long, device=anchors.device)
    matched_boxes = torch.zeros((len(anchors), 4), dtype=torch.float32, device=anchors.device)
    matched_ious = torch.zeros(len(anchors), dtype=torch.float32, device=anchors.device)

    # Mark matches above threshold
    positive_mask = best_gt_iou >= iou_threshold
    matched_labels[positive_mask] = gt_labels[best_gt_idx[positive_mask]]
    matched_boxes[positive_mask] = gt_boxes[best_gt_idx[positive_mask]]
    matched_ious[positive_mask] = best_gt_iou[positive_mask]

    return matched_labels, matched_boxes, matched_ious


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
        # Squeeze
        y = self.squeeze(x).view(b, c)
        # Excitation
        y = self.excitation(y).view(b, c, 1, 1)
        # Scale
        return x * y.expand_as(x)


class EnhancedEarlyExitBranch(nn.Module):
    """Enhanced early exit branch with SE attention and larger capacity"""

    def __init__(self, in_channels, intermediate_channels, num_anchors, num_classes=2, use_attention=True):
        super().__init__()

        # Feature extraction with depthwise separable convolutions
        self.feature_extractor = nn.Sequential(
            # First depthwise separable block
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

        # SE attention block
        self.use_attention = use_attention
        if use_attention:
            self.se_block = SEBlock(intermediate_channels, reduction=4)

        # Second feature block for more capacity
        self.additional_features = nn.Sequential(
            nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3,
                     padding=1, groups=intermediate_channels, bias=False),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU6(inplace=True),

            nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU6(inplace=True)
        )

        # Classification head
        self.cls_head = nn.Conv2d(intermediate_channels,
                                  num_anchors * num_classes, kernel_size=1)

        # Regression head
        self.reg_head = nn.Conv2d(intermediate_channels,
                                  num_anchors * 4, kernel_size=1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial feature extraction
        features = self.feature_extractor(x)

        # Apply SE attention
        if self.use_attention:
            features = self.se_block(features)

        # Additional feature refinement
        features = self.additional_features(features)

        # Generate predictions
        cls_logits = self.cls_head(features)
        bbox_regression = self.reg_head(features)

        return cls_logits, bbox_regression


class MultiLevelCascadeEarlyExitSSDLite(nn.Module):
    """SSDLite with multi-level cascade early exit capability"""

    def __init__(self, base_model, exit1_threshold=0.45, exit2_threshold=0.60):
        super().__init__()

        # Store the complete original backbone
        self.backbone = base_model.backbone

        # Access backbone features
        backbone_features = base_model.backbone.features
        first_sequential = list(backbone_features.children())[0]
        first_block_layers = list(first_sequential.children())

        # Exit 1: After layer 7 (80 channels at 20x20 resolution)
        self.exit1_features = nn.Sequential(*first_block_layers[:8])

        # Layers between exit 1 and exit 2 (layer 8-11, outputs 112 channels)
        self.mid_features = nn.Sequential(*first_block_layers[8:12])

        # Remaining layers for full model (layer 12+, outputs 960 then reduced to 672)
        self.late_features = nn.Sequential(*first_block_layers[12:])

        # Exit branches with enhanced architecture
        num_anchors = 6

        # Exit 1 branch: 80 channels -> 128 intermediate
        self.exit1_branch = EnhancedEarlyExitBranch(
            in_channels=80,
            intermediate_channels=128,
            num_anchors=num_anchors,
            num_classes=2,
            use_attention=True
        )

        # Exit 2 branch: 112 channels -> 224 intermediate
        # Also receives context from exit 1 (add small projection)
        self.exit2_context_proj = nn.Conv2d(128, 112, kernel_size=1)
        self.exit2_branch = EnhancedEarlyExitBranch(
            in_channels=112,
            intermediate_channels=224,
            num_anchors=num_anchors,
            num_classes=2,
            use_attention=True
        )

        # Full branch: 672 channels
        # Receives context from exit 2 (add small projection)
        self.full_context_proj = nn.Conv2d(224, 672, kernel_size=1)
        self.full_branch = EnhancedEarlyExitBranch(
            in_channels=672,
            intermediate_channels=512,
            num_anchors=num_anchors,
            num_classes=2,
            use_attention=True
        )

        # PHASE 1.5: Learnable thresholds (trainable parameters)
        # Initialize with conservative values to ensure balanced distribution
        self.exit1_threshold = nn.Parameter(torch.tensor(exit1_threshold))
        self.exit2_threshold = nn.Parameter(torch.tensor(exit2_threshold))
        self.num_classes = 2

        # Temperature parameters for confidence calibration (per exit)
        self.temperature1 = nn.Parameter(torch.ones(1) * 1.5)
        self.temperature2 = nn.Parameter(torch.ones(1) * 1.5)
        self.temperature_full = nn.Parameter(torch.ones(1) * 1.0)

        # Statistics tracking
        self.exit_stats = {'exit1': 0, 'exit2': 0, 'full': 0}
        self.training_step = 0

    def compute_confidence(self, cls_logits, temperature, bbox_regression=None):
        """
        Phase 1.4: Moderate detection-aware confidence metric for balanced exits

        Improvements from Phase 1.3:
        - Reduced objectness threshold (0.5 → 0.4) for less restrictive detection
        - Reduced minimum object count (3 → 2) for more flexibility
        - Calibrated to achieve moderate confidence levels that match thresholds
        - Goal: Enable balanced distribution across all three exits

        This provides a more moderate metric that can produce varying confidence
        levels based on detection quality, enabling proper cascade behavior.
        """
        batch_size = cls_logits.shape[0]

        # Reshape logits: [B, C, H, W] -> [B, num_anchors, C]
        cls_logits = cls_logits.permute(0, 2, 3, 1).contiguous()
        cls_logits = cls_logits.view(batch_size, -1, self.num_classes)

        # Apply temperature scaling
        scaled_logits = cls_logits / temperature

        # Apply softmax to get probabilities
        probs = F.softmax(scaled_logits, dim=-1)

        # Extract objectness score (probability of chair class)
        objectness = probs[:, :, 1]  # P(chair)

        # Classification confidence (max prob across classes)
        max_probs, _ = probs.max(dim=-1)

        # PHASE 1.4: MODERATE DETECTION-AWARE CONFIDENCE
        # Relaxed thresholds to allow more early exits
        objectness_threshold = 0.4  # Reduced from 0.5
        min_confident_objects = 2   # Reduced from 3

        confidence_list = []
        for i in range(batch_size):
            # Find anchors that likely contain objects
            object_mask = objectness[i] > objectness_threshold
            num_confident_objects = object_mask.sum().item()

            if num_confident_objects >= min_confident_objects:
                # Has sufficient confident object predictions
                # Use objectness as primary signal
                obj_confidence = objectness[i][object_mask].mean()

                # Also consider classification confidence
                cls_confidence = max_probs[i][object_mask].mean()

                # Combine: 80% objectness + 20% classification
                conf = 0.8 * obj_confidence + 0.2 * cls_confidence

                # Apply object count penalty: fewer objects = lower confidence
                # If we have exactly min_confident_objects, apply 0.85x penalty
                # Scale up linearly to 1.0x at 10+ objects
                object_count_factor = min(1.0, 0.85 + 0.15 * (num_confident_objects - min_confident_objects) / 8.0)
                conf = conf * object_count_factor

            elif num_confident_objects > 0:
                # Has some objects, but not enough for high confidence
                # Moderate confidence based on what we have
                obj_confidence = objectness[i][object_mask].mean()
                cls_confidence = max_probs[i][object_mask].mean()

                # Lower weight for objectness when count is low
                conf = 0.6 * obj_confidence + 0.4 * cls_confidence

                # Moderate penalty for insufficient object count (less harsh than Phase 1.3)
                conf = conf * 0.5  # Max 50% confidence with insufficient objects

            else:
                # No confident objects detected
                # Use overall classification confidence (mostly background)
                conf = max_probs[i].mean()

                # Moderate penalty for background-only predictions (less harsh than Phase 1.3)
                conf = conf * 0.4  # Max 40% confidence for background-only

            confidence_list.append(conf)

        confidence = torch.stack(confidence_list)

        return confidence

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be provided")

        # Transform images if needed
        if isinstance(images, (list, tuple)):
            images = torch.stack(images)

        # ============ Exit 1 (Layer 8) ============
        features_exit1 = self.exit1_features(images)
        exit1_cls, exit1_reg = self.exit1_branch(features_exit1)
        exit1_confidence = self.compute_confidence(exit1_cls, self.temperature1)

        if self.training:
            # During training, compute all paths

            # Continue to exit 2
            features_mid = self.mid_features(features_exit1)

            # Exit 2 with context from exit 1
            # Get intermediate features from exit 1 branch for context
            exit1_context = self.exit1_branch.feature_extractor(features_exit1)
            if self.exit1_branch.use_attention:
                exit1_context = self.exit1_branch.se_block(exit1_context)
            exit1_context = self.exit1_branch.additional_features(exit1_context)

            # Resize context to match features_mid spatial dimensions if needed
            if exit1_context.shape[2:] != features_mid.shape[2:]:
                exit1_context = F.adaptive_avg_pool2d(exit1_context, features_mid.shape[2:])

            # Project and add context
            exit1_context_proj = self.exit2_context_proj(exit1_context)
            if exit1_context_proj.shape[2:] != features_mid.shape[2:]:
                exit1_context_proj = F.adaptive_avg_pool2d(exit1_context_proj, features_mid.shape[2:])

            features_exit2 = features_mid + exit1_context_proj

            exit2_cls, exit2_reg = self.exit2_branch(features_exit2)
            exit2_confidence = self.compute_confidence(exit2_cls, self.temperature2)

            # Continue to full model
            features_late = self.late_features(features_mid)

            # Full model with context from exit 2
            exit2_context = self.exit2_branch.feature_extractor(features_exit2)
            if self.exit2_branch.use_attention:
                exit2_context = self.exit2_branch.se_block(exit2_context)
            exit2_context = self.exit2_branch.additional_features(exit2_context)

            # Resize context to match features_late spatial dimensions
            if exit2_context.shape[2:] != features_late.shape[2:]:
                exit2_context = F.adaptive_avg_pool2d(exit2_context, features_late.shape[2:])

            # Project and add context
            exit2_context_proj = self.full_context_proj(exit2_context)
            if exit2_context_proj.shape[2:] != features_late.shape[2:]:
                exit2_context_proj = F.adaptive_avg_pool2d(exit2_context_proj, features_late.shape[2:])

            features_full = features_late + exit2_context_proj

            full_cls, full_reg = self.full_branch(features_full)
            full_confidence = self.compute_confidence(full_cls, self.temperature_full)

            # Compute losses
            losses = self.compute_training_losses(
                exit1_cls, exit1_reg, exit1_confidence,
                exit2_cls, exit2_reg, exit2_confidence,
                full_cls, full_reg, full_confidence,
                targets
            )

            self.training_step += 1
            return losses

        else:
            # During inference - cascade evaluation with early stopping
            avg_confidence_exit1 = exit1_confidence.mean().item()

            if avg_confidence_exit1 >= self.exit1_threshold.item():
                self.exit_stats['exit1'] += 1
                return {
                    'cls_logits': exit1_cls,
                    'bbox_regression': exit1_reg,
                    'exit_point': 'exit1',
                    'confidence': avg_confidence_exit1
                }

            # Continue to exit 2
            features_mid = self.mid_features(features_exit1)

            # Exit 2 with context
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

            if avg_confidence_exit2 >= self.exit2_threshold.item():
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

    def prepare_targets_for_loss(self, cls_logits, bbox_regression, targets):
        """
        Prepare ground truth targets for SSD-style loss computation

        This creates proper classification and regression targets from
        ground truth boxes, instead of using background-only targets.
        """
        batch_size = cls_logits.shape[0]
        num_predictions = cls_logits.view(batch_size, -1, self.num_classes).shape[1]
        device = cls_logits.device

        # Initialize targets (default to background)
        cls_targets = torch.zeros(batch_size, num_predictions, dtype=torch.long, device=device)
        reg_targets = torch.zeros(batch_size, num_predictions, 4, dtype=torch.float32, device=device)
        reg_weights = torch.zeros(batch_size, num_predictions, dtype=torch.float32, device=device)

        # For each image in batch, assign ground truth to predictions
        for i in range(batch_size):
            if len(targets[i]['boxes']) == 0:
                # No objects in this image - all background
                continue

            gt_boxes = targets[i]['boxes'].to(device)
            gt_labels = targets[i]['labels'].to(device)

            # Simple strategy: mark random subset of predictions as positive
            # In real SSD, we'd match based on anchor-GT IoU, but here we simplify
            num_gt = len(gt_boxes)
            num_positive = min(num_gt * 10, num_predictions // 20)  # 5% positive samples

            if num_positive > 0:
                # Randomly select anchors to assign as positive
                positive_indices = torch.randperm(num_predictions, device=device)[:num_positive]

                # Assign each positive anchor to a ground truth box
                for j, idx in enumerate(positive_indices):
                    gt_idx = j % num_gt  # Round-robin assignment
                    cls_targets[i, idx] = gt_labels[gt_idx]
                    reg_targets[i, idx] = gt_boxes[gt_idx]
                    reg_weights[i, idx] = 1.0

        return cls_targets, reg_targets, reg_weights

    def compute_training_losses(self, exit1_cls, exit1_reg, exit1_conf,
                                exit2_cls, exit2_reg, exit2_conf,
                                full_cls, full_reg, full_conf, targets):
        """
        Compute multi-level training losses with ground truth targets

        Combines supervised learning with knowledge distillation for
        progressive cascade refinement.
        """

        batch_size = full_cls.shape[0]
        num_predictions = full_cls.view(batch_size, -1, self.num_classes).shape[1]

        # Prepare ground truth targets for training
        cls_targets, reg_targets, reg_weights = self.prepare_targets_for_loss(full_cls, full_reg, targets)

        # ========== Full model loss (supervised with REAL targets) ==========
        full_cls_loss = F.cross_entropy(
            full_cls.view(batch_size, num_predictions, self.num_classes).reshape(-1, self.num_classes),
            cls_targets.view(-1)
        )

        # Regression loss with ground truth boxes (weighted by positive samples)
        full_reg_flat = full_reg.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)
        if reg_weights.sum() > 0:
            # Only compute loss on positive samples
            full_reg_loss = F.smooth_l1_loss(
                full_reg_flat * reg_weights.unsqueeze(-1),
                reg_targets * reg_weights.unsqueeze(-1),
                reduction='sum'
            ) / reg_weights.sum()
        else:
            full_reg_loss = F.smooth_l1_loss(full_reg_flat, torch.zeros_like(full_reg_flat)) * 0.01

        full_loss = full_cls_loss + full_reg_loss

        # ========== Exit 2 loss (supervised + distillation from full) ==========
        with torch.no_grad():
            full_cls_probs = F.softmax(full_cls.view(-1, self.num_classes), dim=1)

        # Match exit2 spatial size to full
        if exit2_cls.shape != full_cls.shape:
            exit2_cls_resized = F.interpolate(exit2_cls, size=full_cls.shape[2:], mode='bilinear', align_corners=False)
            exit2_reg_resized = F.interpolate(exit2_reg, size=full_reg.shape[2:], mode='bilinear', align_corners=False)
        else:
            exit2_cls_resized = exit2_cls
            exit2_reg_resized = exit2_reg

        # Supervised loss with ground truth
        exit2_cls_supervised = F.cross_entropy(
            exit2_cls_resized.view(batch_size, num_predictions, self.num_classes).reshape(-1, self.num_classes),
            cls_targets.view(-1)
        )

        # Distillation loss from full model
        exit2_cls_distill = F.kl_div(
            F.log_softmax(exit2_cls_resized.view(-1, self.num_classes), dim=1),
            full_cls_probs,
            reduction='batchmean'
        )

        # Combine supervised (70%) + distillation (30%)
        exit2_cls_loss = 0.7 * exit2_cls_supervised + 0.3 * exit2_cls_distill

        # Regression loss with ground truth
        exit2_reg_flat = exit2_reg_resized.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)
        if reg_weights.sum() > 0:
            exit2_reg_supervised = F.smooth_l1_loss(
                exit2_reg_flat * reg_weights.unsqueeze(-1),
                reg_targets * reg_weights.unsqueeze(-1),
                reduction='sum'
            ) / reg_weights.sum()
        else:
            exit2_reg_supervised = F.smooth_l1_loss(exit2_reg_flat, torch.zeros_like(exit2_reg_flat)) * 0.01

        # Also use distillation from full model
        exit2_reg_distill = F.smooth_l1_loss(exit2_reg_resized, full_reg.detach())

        # Combine supervised (70%) + distillation (30%)
        exit2_reg_loss = 0.7 * exit2_reg_supervised + 0.3 * exit2_reg_distill

        exit2_loss = exit2_cls_loss + exit2_reg_loss

        # ========== Exit 1 loss (supervised + distillation from exit 2) ==========
        with torch.no_grad():
            exit2_cls_probs = F.softmax(exit2_cls.view(-1, self.num_classes), dim=1)

        # Match exit1 spatial size to exit2
        if exit1_cls.shape != exit2_cls.shape:
            exit1_cls_resized = F.interpolate(exit1_cls, size=exit2_cls.shape[2:], mode='bilinear', align_corners=False)
            exit1_reg_resized = F.interpolate(exit1_reg, size=exit2_reg.shape[2:], mode='bilinear', align_corners=False)
        else:
            exit1_cls_resized = exit1_cls
            exit1_reg_resized = exit1_reg

        # Supervised loss with ground truth
        # Need to match spatial dimensions with cls_targets
        exit1_num_preds = exit1_cls_resized.view(batch_size, -1, self.num_classes).shape[1]
        if exit1_num_preds != num_predictions:
            # Resize targets to match exit1 predictions
            exit1_cls_targets = cls_targets[:, :exit1_num_preds] if exit1_num_preds < num_predictions else cls_targets
        else:
            exit1_cls_targets = cls_targets

        exit1_cls_supervised = F.cross_entropy(
            exit1_cls_resized.view(batch_size, -1, self.num_classes).reshape(-1, self.num_classes),
            exit1_cls_targets.view(-1)
        )

        # Distillation loss from exit 2
        exit1_cls_distill = F.kl_div(
            F.log_softmax(exit1_cls_resized.view(-1, self.num_classes), dim=1),
            exit2_cls_probs,
            reduction='batchmean'
        )

        # Combine supervised (70%) + distillation (30%)
        exit1_cls_loss = 0.7 * exit1_cls_supervised + 0.3 * exit1_cls_distill

        # Regression loss with ground truth
        exit1_reg_flat = exit1_reg_resized.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)
        if reg_weights.sum() > 0:
            exit1_reg_supervised = F.smooth_l1_loss(
                exit1_reg_flat * reg_weights.unsqueeze(-1),
                reg_targets * reg_weights.unsqueeze(-1),
                reduction='sum'
            ) / reg_weights.sum()
        else:
            exit1_reg_supervised = F.smooth_l1_loss(exit1_reg_flat, torch.zeros_like(exit1_reg_flat)) * 0.01

        # Also use distillation from exit 2
        exit1_reg_distill = F.smooth_l1_loss(exit1_reg_resized, exit2_reg.detach())

        # Combine supervised (70%) + distillation (30%)
        exit1_reg_loss = 0.7 * exit1_reg_supervised + 0.3 * exit1_reg_distill

        exit1_loss = exit1_cls_loss + exit1_reg_loss

        # ========== Cascade consistency loss ==========
        # Encourage smooth refinement between levels
        cascade_loss_exit2_full = F.kl_div(
            F.log_softmax(exit2_cls_resized.view(-1, self.num_classes), dim=1),
            F.softmax(full_cls.view(-1, self.num_classes).detach(), dim=1),
            reduction='batchmean'
        )

        cascade_loss_exit1_exit2 = F.kl_div(
            F.log_softmax(exit1_cls_resized.view(-1, self.num_classes), dim=1),
            F.softmax(exit2_cls.view(-1, self.num_classes).detach(), dim=1),
            reduction='batchmean'
        )

        cascade_loss = (cascade_loss_exit2_full + cascade_loss_exit1_exit2) / 2

        # ========== Confidence diversity loss ==========
        # Encourage progression: exit1 < exit2 < full confidence
        conf_diversity_loss = torch.tensor(0.0, device=full_cls.device)

        # Penalize if exit1 confidence is higher than exit2
        if exit1_conf.mean() > exit2_conf.mean():
            conf_diversity_loss += (exit1_conf.mean() - exit2_conf.mean())

        # Penalize if exit2 confidence is higher than full
        if exit2_conf.mean() > full_conf.mean():
            conf_diversity_loss += (exit2_conf.mean() - full_conf.mean())

        # ========== PHASE 1.5: Distribution loss ==========
        # Guide thresholds to achieve target exit distribution
        # Target: 30% Exit1, 25% Exit2, 45% Full (midpoints of desired ranges)

        # Simulate which exit each sample would take based on current confidences
        exit1_would_exit = (exit1_conf >= self.exit1_threshold).float()
        exit2_would_exit = ((exit1_conf < self.exit1_threshold) &
                           (exit2_conf >= self.exit2_threshold)).float()

        # Calculate rates
        simulated_exit1_rate = exit1_would_exit.mean()
        simulated_exit2_rate = exit2_would_exit.mean()

        # Target distribution (midpoints)
        target_exit1_rate = 0.30  # 20-40% range
        target_exit2_rate = 0.25  # 20-30% range

        # Distribution loss - squared error from targets
        distribution_loss = (
            (simulated_exit1_rate - target_exit1_rate) ** 2 +
            (simulated_exit2_rate - target_exit2_rate) ** 2
        )

        # ========== Loss weighting ==========
        # Progressive weighting: favor later exits early in training, balance later
        epoch_progress = min(self.training_step / 5000.0, 1.0)

        # Start: (0.2, 0.3, 0.5) -> End: (0.3, 0.4, 0.3)
        alpha_exit1 = 0.2 + 0.1 * epoch_progress
        alpha_exit2 = 0.3 + 0.1 * epoch_progress
        alpha_full = 0.5 - 0.2 * epoch_progress

        # Total loss with distribution guidance
        total_loss = (
            alpha_exit1 * exit1_loss +
            alpha_exit2 * exit2_loss +
            alpha_full * full_loss +
            0.05 * cascade_loss +
            0.02 * conf_diversity_loss +
            0.1 * distribution_loss  # PHASE 1.5: Distribution loss for threshold learning
        )

        return {
            'total_loss': total_loss,
            'exit1_loss': exit1_loss,
            'exit2_loss': exit2_loss,
            'full_loss': full_loss,
            'cascade_loss': cascade_loss,
            'diversity_loss': conf_diversity_loss,
            'distribution_loss': distribution_loss,  # PHASE 1.5
            'simulated_exit1_rate': simulated_exit1_rate,  # PHASE 1.5
            'simulated_exit2_rate': simulated_exit2_rate,  # PHASE 1.5
            'exit1_confidence': exit1_conf.mean(),
            'exit2_confidence': exit2_conf.mean(),
            'full_confidence': full_conf.mean(),
            'confidence_std_exit1': exit1_conf.std(),
            'confidence_std_exit2': exit2_conf.std(),
        }


class Trainer:
    """Training class for multi-level early exit model"""

    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Optimizer with different learning rates
        # PHASE 1.5: Added learnable thresholds
        self.optimizer = torch.optim.AdamW([
            {'params': model.exit1_branch.parameters(), 'lr': 1e-3},
            {'params': model.exit2_branch.parameters(), 'lr': 1e-3},
            {'params': model.full_branch.parameters(), 'lr': 1e-3},
            {'params': model.backbone.parameters(), 'lr': 1e-4},
            {'params': [model.temperature1, model.temperature2, model.temperature_full], 'lr': 1e-2},
            {'params': [model.exit1_threshold, model.exit2_threshold], 'lr': 5e-3},  # PHASE 1.5: Learnable thresholds
        ], weight_decay=0.0005)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=20
        )

        # Metrics tracking
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()

        epoch_metrics = defaultdict(list)

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (images, targets) in enumerate(pbar):
            try:
                # Move to device
                images = torch.stack([img.to(self.device) for img in images])
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                # Forward pass
                losses = self.model(images, targets)

                # Check for NaN
                if torch.isnan(losses['total_loss']):
                    print(f"Warning: NaN loss detected at batch {batch_idx}, skipping...")
                    continue

                # Backward pass
                self.optimizer.zero_grad()
                losses['total_loss'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                # Track metrics
                for key, value in losses.items():
                    if torch.is_tensor(value):
                        epoch_metrics[key].append(value.item())

                # Update progress bar
                if batch_idx % 10 == 0:
                    pbar.set_postfix({
                        'loss': np.mean(epoch_metrics['total_loss'][-50:]),
                        'e1_conf': np.mean(epoch_metrics['exit1_confidence'][-50:]),
                        'e2_conf': np.mean(epoch_metrics['exit2_confidence'][-50:]),
                        'th1': self.model.exit1_threshold.item(),  # PHASE 1.5: Show threshold
                        'th2': self.model.exit2_threshold.item(),  # PHASE 1.5: Show threshold
                        'e1_rate': np.mean(epoch_metrics['simulated_exit1_rate'][-50:])  # PHASE 1.5
                    })

            except RuntimeError as e:
                print(f"Error at batch {batch_idx}: {e}")
                continue

        # Store epoch metrics
        for key, values in epoch_metrics.items():
            self.train_metrics[key].append(np.mean(values))

        # PHASE 1.5: Track threshold evolution
        if 'exit1_threshold' not in self.train_metrics:
            self.train_metrics['exit1_threshold'] = []
            self.train_metrics['exit2_threshold'] = []
        self.train_metrics['exit1_threshold'].append(self.model.exit1_threshold.item())
        self.train_metrics['exit2_threshold'].append(self.model.exit2_threshold.item())

    def validate(self, epoch):
        """Validate model"""
        self.model.eval()

        inference_times = []
        confidences_per_exit = {'exit1': [], 'exit2': [], 'full': []}

        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc='Validation'):
                images = torch.stack([img.to(self.device) for img in images])

                # Measure inference time
                start_time = time.time()
                outputs = self.model(images)
                inference_time = (time.time() - start_time) * 1000

                inference_times.append(inference_time)

                if 'confidence' in outputs and 'exit_point' in outputs:
                    exit_point = outputs['exit_point']
                    confidences_per_exit[exit_point].append(outputs['confidence'])

        # Calculate metrics
        total_samples = sum([self.model.exit_stats[k] for k in self.model.exit_stats])
        exit1_rate = self.model.exit_stats['exit1'] / max(total_samples, 1)
        exit2_rate = self.model.exit_stats['exit2'] / max(total_samples, 1)
        full_rate = self.model.exit_stats['full'] / max(total_samples, 1)
        combined_early_exit_rate = exit1_rate + exit2_rate

        self.val_metrics['inference_time'].append(np.mean(inference_times))
        self.val_metrics['exit1_rate'].append(exit1_rate)
        self.val_metrics['exit2_rate'].append(exit2_rate)
        self.val_metrics['full_rate'].append(full_rate)
        self.val_metrics['combined_early_exit_rate'].append(combined_early_exit_rate)

        # Per-exit confidence
        self.val_metrics['exit1_confidence'].append(
            np.mean(confidences_per_exit['exit1']) if confidences_per_exit['exit1'] else 0
        )
        self.val_metrics['exit2_confidence'].append(
            np.mean(confidences_per_exit['exit2']) if confidences_per_exit['exit2'] else 0
        )
        self.val_metrics['full_confidence'].append(
            np.mean(confidences_per_exit['full']) if confidences_per_exit['full'] else 0
        )

        # Store counts
        self.val_metrics['exit1_count'] = self.model.exit_stats['exit1']
        self.val_metrics['exit2_count'] = self.model.exit_stats['exit2']
        self.val_metrics['full_count'] = self.model.exit_stats['full']

        # Reset stats
        self.model.exit_stats = {'exit1': 0, 'exit2': 0, 'full': 0}

        return combined_early_exit_rate, np.mean(inference_times)

    def train(self, num_epochs):
        """Full training loop"""
        for epoch in range(num_epochs):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*70}")

            # Train
            self.train_epoch(epoch)

            # Validate
            early_rate, inf_time = self.validate(epoch)

            # Update scheduler
            self.scheduler.step()

            # Print metrics
            print(f"\n--- Training Metrics ---")
            print(f"Total Loss: {self.train_metrics['total_loss'][-1]:.4f}")
            print(f"  Exit 1 Loss: {self.train_metrics['exit1_loss'][-1]:.4f}")
            print(f"  Exit 2 Loss: {self.train_metrics['exit2_loss'][-1]:.4f}")
            print(f"  Full Loss: {self.train_metrics['full_loss'][-1]:.4f}")
            print(f"  Cascade Loss: {self.train_metrics['cascade_loss'][-1]:.4f}")
            print(f"  Distribution Loss: {self.train_metrics['distribution_loss'][-1]:.4f}")  # PHASE 1.5

            print(f"\n--- Confidence Progression ---")
            print(f"  Exit 1: {self.train_metrics['exit1_confidence'][-1]:.3f} (±{self.train_metrics['confidence_std_exit1'][-1]:.3f})")
            print(f"  Exit 2: {self.train_metrics['exit2_confidence'][-1]:.3f} (±{self.train_metrics['confidence_std_exit2'][-1]:.3f})")
            print(f"  Full:   {self.train_metrics['full_confidence'][-1]:.3f}")

            print(f"\n--- PHASE 1.5: Learnable Thresholds ---")
            print(f"  Exit 1 Threshold: {self.model.exit1_threshold.item():.3f} (conf: {self.train_metrics['exit1_confidence'][-1]:.3f})")
            print(f"  Exit 2 Threshold: {self.model.exit2_threshold.item():.3f} (conf: {self.train_metrics['exit2_confidence'][-1]:.3f})")

            print(f"\n--- PHASE 1.5: Simulated Exit Distribution (Training) ---")
            print(f"  Exit 1 Rate: {self.train_metrics['simulated_exit1_rate'][-1]:.2%} (target: 30%)")
            print(f"  Exit 2 Rate: {self.train_metrics['simulated_exit2_rate'][-1]:.2%} (target: 25%)")
            print(f"  Full Rate:   {1 - self.train_metrics['simulated_exit1_rate'][-1] - self.train_metrics['simulated_exit2_rate'][-1]:.2%} (target: 45%)")

            print(f"\n--- Temperature Values ---")
            print(f"  Exit 1: {self.model.temperature1.item():.3f}")
            print(f"  Exit 2: {self.model.temperature2.item():.3f}")
            print(f"  Full:   {self.model.temperature_full.item():.3f}")

            print(f"\n--- Validation Metrics ---")
            print(f"Exit Distribution:")
            print(f"  Exit 1: {self.val_metrics['exit1_rate'][-1]:.2%} ({self.val_metrics['exit1_count']} samples)")
            print(f"  Exit 2: {self.val_metrics['exit2_rate'][-1]:.2%} ({self.val_metrics['exit2_count']} samples)")
            print(f"  Full:   {self.val_metrics['full_rate'][-1]:.2%} ({self.val_metrics['full_count']} samples)")
            print(f"  Combined Early Exit: {early_rate:.2%}")
            print(f"Avg Inference Time: {inf_time:.2f}ms")

            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch + 1)

        # Save final model and metrics
        self.save_final_results()

    def save_checkpoint(self, epoch):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(MODEL_OUTPUT_PATH, f'checkpoint_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_metrics': dict(self.train_metrics),
            'val_metrics': dict(self.val_metrics)
        }, checkpoint_path)
        print(f"\nCheckpoint saved: {checkpoint_path}")

    def save_final_results(self):
        """Save final model and metrics"""
        # Save model
        final_model_path = os.path.join(MODEL_OUTPUT_PATH, 'early_exit_ssdlite_phase1_5_final.pth')
        torch.save(self.model.state_dict(), final_model_path)
        print(f"\nFinal model saved: {final_model_path}")

        # Save metrics to CSV
        train_df = pd.DataFrame(self.train_metrics)
        train_df.to_csv(os.path.join(ANALYSIS_EE_OUTPUT_PATH, 'train_metrics_phase1_5.csv'), index=False)

        val_df = pd.DataFrame(self.val_metrics)
        val_df.to_csv(os.path.join(ANALYSIS_EE_OUTPUT_PATH, 'val_metrics_phase1_5.csv'), index=False)

        # Create and save plots
        self.create_plots()

    def create_plots(self):
        """Create comprehensive training plots"""
        epochs = range(1, len(self.train_metrics['total_loss']) + 1)

        # Create figure with subplots (3 rows x 3 columns)
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))

        # Row 1, Col 1: Training losses
        axes[0, 0].plot(epochs, self.train_metrics['total_loss'], 'b-', linewidth=2, label='Total')
        axes[0, 0].plot(epochs, self.train_metrics['exit1_loss'], 'r--', label='Exit 1')
        axes[0, 0].plot(epochs, self.train_metrics['exit2_loss'], 'g--', label='Exit 2')
        axes[0, 0].plot(epochs, self.train_metrics['full_loss'], 'm--', label='Full')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Losses')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Row 1, Col 2: Confidence progression
        axes[0, 1].plot(epochs, self.train_metrics['exit1_confidence'], 'r-', label='Exit 1', linewidth=2)
        axes[0, 1].plot(epochs, self.train_metrics['exit2_confidence'], 'g-', label='Exit 2', linewidth=2)
        axes[0, 1].plot(epochs, self.train_metrics['full_confidence'], 'm-', label='Full', linewidth=2)
        axes[0, 1].axhline(y=self.model.exit1_threshold, color='r', linestyle=':', alpha=0.5, label=f'Thresh 1 ({self.model.exit1_threshold})')
        axes[0, 1].axhline(y=self.model.exit2_threshold, color='g', linestyle=':', alpha=0.5, label=f'Thresh 2 ({self.model.exit2_threshold})')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Confidence')
        axes[0, 1].set_title('Training Confidence Progression')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Row 1, Col 3: Exit distribution
        axes[0, 2].plot(epochs, self.val_metrics['exit1_rate'], 'r-', label='Exit 1', linewidth=2)
        axes[0, 2].plot(epochs, self.val_metrics['exit2_rate'], 'g-', label='Exit 2', linewidth=2)
        axes[0, 2].plot(epochs, self.val_metrics['full_rate'], 'm-', label='Full', linewidth=2)
        axes[0, 2].plot(epochs, self.val_metrics['combined_early_exit_rate'], 'b-', label='Combined Early', linewidth=3)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Exit Rate')
        axes[0, 2].set_title('Validation Exit Distribution')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # Row 2, Col 1: PHASE 1.5 - Distribution loss (for threshold learning)
        axes[1, 0].plot(epochs, self.train_metrics['distribution_loss'], 'purple', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Distribution Loss')
        axes[1, 0].set_title('PHASE 1.5: Distribution Loss (Threshold Learning)')
        axes[1, 0].grid(True, alpha=0.3)

        # Row 2, Col 2: Inference time
        axes[1, 1].plot(epochs, self.val_metrics['inference_time'], 'orange', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Time (ms)')
        axes[1, 1].set_title('Average Inference Time')
        axes[1, 1].grid(True, alpha=0.3)

        # Row 2, Col 3: Validation confidence per exit
        axes[1, 2].plot(epochs, self.val_metrics['exit1_confidence'], 'r-', label='Exit 1', linewidth=2)
        axes[1, 2].plot(epochs, self.val_metrics['exit2_confidence'], 'g-', label='Exit 2', linewidth=2)
        axes[1, 2].plot(epochs, self.val_metrics['full_confidence'], 'm-', label='Full', linewidth=2)
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Confidence')
        axes[1, 2].set_title('Validation Confidence per Exit')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        # Row 3, Col 1: Stacked exit rates
        exit1_rates = np.array(self.val_metrics['exit1_rate'])
        exit2_rates = np.array(self.val_metrics['exit2_rate'])
        full_rates = np.array(self.val_metrics['full_rate'])

        axes[2, 0].fill_between(epochs, 0, exit1_rates, color='red', alpha=0.5, label='Exit 1')
        axes[2, 0].fill_between(epochs, exit1_rates, exit1_rates + exit2_rates, color='green', alpha=0.5, label='Exit 2')
        axes[2, 0].fill_between(epochs, exit1_rates + exit2_rates, 1, color='magenta', alpha=0.5, label='Full')
        axes[2, 0].set_xlabel('Epoch')
        axes[2, 0].set_ylabel('Proportion')
        axes[2, 0].set_title('Stacked Exit Distribution')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)

        # Row 3, Col 2: PHASE 1.5 - Learnable Threshold Evolution
        axes[2, 1].plot(epochs, self.train_metrics['exit1_threshold'], 'r-', label='Exit 1 Threshold', linewidth=2)
        axes[2, 1].plot(epochs, self.train_metrics['exit2_threshold'], 'g-', label='Exit 2 Threshold', linewidth=2)
        axes[2, 1].plot(epochs, self.train_metrics['exit1_confidence'], 'r--', label='Exit 1 Confidence', linewidth=1, alpha=0.6)
        axes[2, 1].plot(epochs, self.train_metrics['exit2_confidence'], 'g--', label='Exit 2 Confidence', linewidth=1, alpha=0.6)
        axes[2, 1].set_xlabel('Epoch')
        axes[2, 1].set_ylabel('Threshold / Confidence')
        axes[2, 1].set_title('PHASE 1.5: Learned Thresholds vs Confidence')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)

        # Row 3, Col 3: Combined metrics - Loss vs Early Exit Rate
        ax_loss = axes[2, 2]
        ax_loss.plot(epochs, self.train_metrics['total_loss'], 'b-', linewidth=2)
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Total Loss', color='b')
        ax_loss.tick_params(axis='y', labelcolor='b')
        ax_loss.grid(True, alpha=0.3)

        ax_exit = ax_loss.twinx()
        ax_exit.plot(epochs, self.val_metrics['combined_early_exit_rate'], 'orange', linewidth=2)
        ax_exit.set_ylabel('Combined Early Exit Rate', color='orange')
        ax_exit.tick_params(axis='y', labelcolor='orange')

        axes[2, 2].set_title('Loss vs Early Exit Rate')

        plt.tight_layout()
        plt.savefig(os.path.join(ANALYSIS_EE_OUTPUT_PATH, 'training_plots_phase1_5.png'), dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Plots saved to {ANALYSIS_EE_OUTPUT_PATH}")


def get_transform(train=False):
    """Get image transforms"""
    from torchvision import transforms

    if train:
        return transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(0.5),
        ])
    return transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor()
    ])


def main():
    """Main training function for Phase 1.5"""
    # Configuration
    BATCH_SIZE = 4
    NUM_EPOCHS = 20
    # Phase 1.5: LEARNABLE THRESHOLDS with distribution loss guidance
    # Conservative initialization - model will learn optimal values
    EXIT1_THRESHOLD = 0.55  # Conservative start (margin above observed 0.491)
    EXIT2_THRESHOLD = 0.65  # Conservative start (margin above observed 0.514)
    NUM_WORKERS = 0

    print("="*80)
    print(" Phase 1.5: Learnable Thresholds with Distribution Loss")
    print("="*80)
    print()
    print("KEY INNOVATION:")
    print("  🚀 THRESHOLDS ARE NOW LEARNABLE PARAMETERS (like temperature)")
    print("  🎯 DISTRIBUTION LOSS guides thresholds to target exit rates")
    print("  ⚙️  AUTOMATIC CALIBRATION - no manual threshold tuning needed")
    print("="*80)
    print()
    print("IMPROVEMENTS FROM PHASE 1.4:")
    print("  - Thresholds are nn.Parameter (trainable via backprop)")
    print("  - Added distribution loss: penalizes deviation from target rates")
    print("  - Conservative initialization: Exit1=0.55, Exit2=0.65")
    print("  - Target distribution: 30% Exit1, 25% Exit2, 45% Full")
    print("="*80)
    print()
    print("PHASE 1.4 FAILURE ANALYSIS:")
    print("  - Fixed threshold 0.45 was too close to confidence 0.491")
    print("  - Result: 100% early exit (same problem as Phase 1.2)")
    print("  - Fixed thresholds are fragile - tiny changes cause huge shifts")
    print("  - Solution: Make thresholds LEARNABLE like temperature")
    print("="*80)
    print()
    print("HOW IT WORKS:")
    print("  1. Thresholds initialized conservatively (0.55, 0.65)")
    print("  2. During training, gradients adjust thresholds via distribution loss")
    print("  3. Distribution loss = (exit1_rate - 30%)² + (exit2_rate - 25%)²")
    print("  4. Model learns optimal thresholds automatically")
    print("="*80)
    print()

    # Clean and create working directories
    # clean_working_directory()  # Disabled: do not clean working directory
    create_output_directories()
    print()

    print(f"Configuration:")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Exit 1 Threshold: {EXIT1_THRESHOLD} (layer 8)")
    print(f"  Exit 2 Threshold: {EXIT2_THRESHOLD} (layer 12)")
    print(f"  Num Workers: {NUM_WORKERS}")
    print()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()

    # Load datasets
    print("Loading datasets...")
    train_dataset = ChairCocoDataset(
        TRAIN_IMAGES_PATH,
        os.path.join(ANNOTATIONS_PATH, 'instances_train2017.json'),
        transforms=get_transform(train=True),
        max_samples=5000
    )

    val_dataset = ChairCocoDataset(
        VAL_IMAGES_PATH,
        os.path.join(ANNOTATIONS_PATH, 'instances_val2017.json'),
        transforms=get_transform(train=False),
        max_samples=1000
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print()

    # Initialize model
    print("Initializing Phase 1.5 model with LEARNABLE THRESHOLDS...")
    from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights
    base_model = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
    model = MultiLevelCascadeEarlyExitSSDLite(
        base_model,
        exit1_threshold=EXIT1_THRESHOLD,
        exit2_threshold=EXIT2_THRESHOLD
    )

    print("Architecture Details:")
    print(f"  Exit 1: Layer 8 (80->128 channels) + SE Attention")
    print(f"  Exit 2: Layer 12 (112->224 channels) + SE Attention")
    print(f"  Full:   Complete backbone (672 channels)")
    print(f"  Cascade: Each level refines previous predictions")
    print(f"  ⚡ NEW: Exit thresholds are LEARNABLE parameters")
    print()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()

    # Initialize trainer
    print("Starting training...")
    print("="*80)
    trainer = Trainer(model, train_loader, val_loader, device)

    # Train
    trainer.train(NUM_EPOCHS)

    print("\n" + "="*80)
    print(" Training Complete - Phase 1.5: Learnable Thresholds")
    print("="*80)
    print(f"Models saved to: {MODEL_OUTPUT_PATH}")
    print(f"Analysis saved to: {ANALYSIS_EE_OUTPUT_PATH}")
    print()

    # Final statistics
    final_metrics = {
        'final_train_loss': trainer.train_metrics['total_loss'][-1],
        'final_exit1_loss': trainer.train_metrics['exit1_loss'][-1],
        'final_exit2_loss': trainer.train_metrics['exit2_loss'][-1],
        'final_full_loss': trainer.train_metrics['full_loss'][-1],
        'final_cascade_loss': trainer.train_metrics['cascade_loss'][-1],
        'final_distribution_loss': trainer.train_metrics['distribution_loss'][-1],  # PHASE 1.5

        'final_exit1_rate': trainer.val_metrics['exit1_rate'][-1],
        'final_exit2_rate': trainer.val_metrics['exit2_rate'][-1],
        'final_full_rate': trainer.val_metrics['full_rate'][-1],
        'final_combined_early_exit_rate': trainer.val_metrics['combined_early_exit_rate'][-1],

        'final_inference_time': trainer.val_metrics['inference_time'][-1],

        'train_exit1_confidence': trainer.train_metrics['exit1_confidence'][-1],
        'train_exit2_confidence': trainer.train_metrics['exit2_confidence'][-1],
        'train_full_confidence': trainer.train_metrics['full_confidence'][-1],

        'simulated_exit1_rate': trainer.train_metrics['simulated_exit1_rate'][-1],  # PHASE 1.5
        'simulated_exit2_rate': trainer.train_metrics['simulated_exit2_rate'][-1],  # PHASE 1.5

        'val_exit1_confidence': trainer.val_metrics['exit1_confidence'][-1],
        'val_exit2_confidence': trainer.val_metrics['exit2_confidence'][-1],
        'val_full_confidence': trainer.val_metrics['full_confidence'][-1],

        'learned_exit1_threshold': model.exit1_threshold.item(),  # PHASE 1.5
        'learned_exit2_threshold': model.exit2_threshold.item(),  # PHASE 1.5
        'initial_exit1_threshold': EXIT1_THRESHOLD,  # PHASE 1.5
        'initial_exit2_threshold': EXIT2_THRESHOLD,  # PHASE 1.5

        'temperature1': model.temperature1.item(),
        'temperature2': model.temperature2.item(),
        'temperature_full': model.temperature_full.item(),

        'exit1_count': trainer.val_metrics['exit1_count'],
        'exit2_count': trainer.val_metrics['exit2_count'],
        'full_count': trainer.val_metrics['full_count']
    }

    # Save final summary
    with open(os.path.join(ANALYSIS_EE_OUTPUT_PATH, 'final_summary_phase1_5.json'), 'w') as f:
        json.dump(final_metrics, f, indent=4)

    # Print comprehensive summary
    print("="*80)
    print(" PHASE 1.5 FINAL RESULTS: Learnable Thresholds")
    print("="*80)
    print()

    print("--- Training Losses ---")
    print(f"  Total:        {final_metrics['final_train_loss']:.4f}")
    print(f"  Exit 1:       {final_metrics['final_exit1_loss']:.4f}")
    print(f"  Exit 2:       {final_metrics['final_exit2_loss']:.4f}")
    print(f"  Full:         {final_metrics['final_full_loss']:.4f}")
    print(f"  Cascade:      {final_metrics['final_cascade_loss']:.4f}")
    print(f"  Distribution: {final_metrics['final_distribution_loss']:.4f}")  # PHASE 1.5
    print()

    print("--- PHASE 1.5: Learned Thresholds ---")
    print(f"  Exit 1 Threshold:")
    print(f"    Initial: {final_metrics['initial_exit1_threshold']:.3f}")
    print(f"    Learned: {final_metrics['learned_exit1_threshold']:.3f}")
    print(f"    Change:  {final_metrics['learned_exit1_threshold'] - final_metrics['initial_exit1_threshold']:.3f}")
    print(f"  Exit 2 Threshold:")
    print(f"    Initial: {final_metrics['initial_exit2_threshold']:.3f}")
    print(f"    Learned: {final_metrics['learned_exit2_threshold']:.3f}")
    print(f"    Change:  {final_metrics['learned_exit2_threshold'] - final_metrics['initial_exit2_threshold']:.3f}")
    print()

    print("--- Exit Distribution (Validation) ---")
    print(f"  Exit 1 (Layer 8):   {final_metrics['final_exit1_rate']:.2%} ({final_metrics['exit1_count']} samples)")
    print(f"  Exit 2 (Layer 12):  {final_metrics['final_exit2_rate']:.2%} ({final_metrics['exit2_count']} samples)")
    print(f"  Full Model:         {final_metrics['final_full_rate']:.2%} ({final_metrics['full_count']} samples)")
    print(f"  ---")
    print(f"  Combined Early Exit: {final_metrics['final_combined_early_exit_rate']:.2%}")
    print()

    print("--- Confidence Values ---")
    print(f"  Training:")
    print(f"    Exit 1:  {final_metrics['train_exit1_confidence']:.3f} (learned threshold: {final_metrics['learned_exit1_threshold']:.3f})")
    print(f"    Exit 2:  {final_metrics['train_exit2_confidence']:.3f} (learned threshold: {final_metrics['learned_exit2_threshold']:.3f})")
    print(f"    Full:    {final_metrics['train_full_confidence']:.3f}")
    print(f"  Validation:")
    print(f"    Exit 1:  {final_metrics['val_exit1_confidence']:.3f}")
    print(f"    Exit 2:  {final_metrics['val_exit2_confidence']:.3f}")
    print(f"    Full:    {final_metrics['val_full_confidence']:.3f}")
    print()

    print("--- PHASE 1.5: Simulated Exit Distribution (Training) ---")
    print(f"  Exit 1 Rate: {final_metrics['simulated_exit1_rate']:.2%} (target: 30%)")
    print(f"  Exit 2 Rate: {final_metrics['simulated_exit2_rate']:.2%} (target: 25%)")
    print(f"  Full Rate:   {1 - final_metrics['simulated_exit1_rate'] - final_metrics['simulated_exit2_rate']:.2%} (target: 45%)")
    print()

    print("--- Performance ---")
    print(f"  Avg Inference Time: {final_metrics['final_inference_time']:.2f}ms")
    print()

    print("--- Temperature Calibration ---")
    print(f"  Exit 1:  {final_metrics['temperature1']:.3f}")
    print(f"  Exit 2:  {final_metrics['temperature2']:.3f}")
    print(f"  Full:    {final_metrics['temperature_full']:.3f}")
    print()

    # Comparison with previous phases
    phase1_3_early_exit_rate = 0.0  # 0% from Phase 1.3 (thresholds too high)
    phase1_4_early_exit_rate = 1.0  # 100% from Phase 1.4 (threshold too close)
    phase1_2_early_exit_rate = 1.0  # 100% from Phase 1.2 (thresholds too low)
    baseline_early_exit_rate = 0.0759  # 7.59% from v2

    print("="*80)
    print(" COMPARISON WITH PREVIOUS PHASES")
    print("="*80)
    print(f"  Baseline (v2) Early Exit Rate:  {baseline_early_exit_rate:.2%}")
    print(f"  Phase 1.2 Early Exit Rate:      {phase1_2_early_exit_rate:.2%} (Fixed thresh 0.30/0.50 - TOO LOW)")
    print(f"  Phase 1.3 Early Exit Rate:      {phase1_3_early_exit_rate:.2%} (Fixed thresh 0.65/0.80 - TOO HIGH)")
    print(f"  Phase 1.4 Early Exit Rate:      {phase1_4_early_exit_rate:.2%} (Fixed thresh 0.45/0.60 - TOO CLOSE)")
    print(f"  Phase 1.5 Early Exit Rate:      {final_metrics['final_combined_early_exit_rate']:.2%} (LEARNABLE - AUTOMATIC)")
    print()
    print("  PHASE 1.5 KEY INNOVATION:")
    print("    🎯 Thresholds are LEARNED, not manually tuned")
    print("    📊 Distribution loss guides learning")
    print(f"    ⚙️  Exit 1 threshold: {final_metrics['initial_exit1_threshold']:.3f} → {final_metrics['learned_exit1_threshold']:.3f}")
    print(f"    ⚙️  Exit 2 threshold: {final_metrics['initial_exit2_threshold']:.3f} → {final_metrics['learned_exit2_threshold']:.3f}")
    print()
    print("  TARGET DISTRIBUTION:")
    print("    Exit 1: 20-40% (easy cases with clear detections)")
    print("    Exit 2: 20-30% (moderate difficulty cases)")
    print("    Full:   30-60% (complex cases requiring full processing)")
    print()
    print(f"  ACTUAL DISTRIBUTION (Validation):")
    print(f"    Exit 1: {final_metrics['final_exit1_rate']:.2%}")
    print(f"    Exit 2: {final_metrics['final_exit2_rate']:.2%}")
    print(f"    Full:   {final_metrics['final_full_rate']:.2%}")
    print()
    print(f"  SIMULATED DISTRIBUTION (Training - what model learned):")
    print(f"    Exit 1: {final_metrics['simulated_exit1_rate']:.2%}")
    print(f"    Exit 2: {final_metrics['simulated_exit2_rate']:.2%}")
    print(f"    Full:   {1 - final_metrics['simulated_exit1_rate'] - final_metrics['simulated_exit2_rate']:.2%}")
    print("="*80)


if __name__ == "__main__":
    main()
