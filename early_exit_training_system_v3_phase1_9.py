# early_exit_training_system_v3_phase1_9.py
# Phase 1.9: BALANCED Approach - Moderate Curriculum + Relaxed Gates + Warmup
#
# Phase 1.8 Failure Analysis:
# - Exit1/Exit2 confidence collapsed to 0.000 (quality gate too strict)
# - 0% early exits (100% Full model - defeats purpose)
# - MIN_DETECTIONS=3 + FG_THRESHOLD=0.70 too prohibitive
# - Curriculum too high: 0.40→0.65, 0.50→0.75
# - Temperature clamping WORKS ✅ (1.55, 2.15 within 0.5-3.0)
#
# Phase 1.9 Solutions - BALANCED STRICTNESS:
# 1. MODERATE CURRICULUM: 0.35→0.55 (exit1), 0.45→0.65 (exit2)
#    - Between Phase 1.7's too-low and Phase 1.8's too-high
# 2. RELAXED QUALITY GATES: MIN_DETECTIONS=2, FG_THRESHOLD=0.60
#    - Early layers have less capacity, shouldn't need full-model standards
# 3. WARMUP PERIOD: Epochs 1-3 full-only, 4-6 gradual, 7-20 full cascade
#    - Let early exits learn features before requiring exits
# 4. SMOOTH CONFIDENCE GATING: Sigmoid instead of binary pass/fail
#    - Gradual penalty instead of hard rejection
# 5. KEEP WHAT WORKS: Temperature clamping, foreground-aware confidence

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


def gumbel_softmax_sample(logits, temperature=1.0, hard=True):
    """
    Gumbel-Softmax sampling for discrete decisions with gradients

    Args:
        logits: [batch_size, num_classes] unnormalized log probabilities
        temperature: controls sharpness (lower = more discrete)
        hard: if True, returns one-hot; if False, returns soft probabilities

    Returns:
        samples: discrete samples (if hard=True) with gradients through Gumbel-Softmax
    """
    # Sample from Gumbel(0, 1)
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)

    # Add Gumbel noise and apply softmax with temperature
    y = F.softmax((logits + gumbel_noise) / temperature, dim=-1)

    if hard:
        # Straight-through estimator: forward pass uses argmax, backward uses soft
        _, max_indices = y.max(dim=-1, keepdim=True)
        y_hard = torch.zeros_like(y).scatter_(-1, max_indices, 1.0)
        y = (y_hard - y).detach() + y  # Gradient flows through soft y

    return y


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
    """SSDLite with multi-level cascade early exit capability - Phase 1.9"""

    def __init__(self, base_model, exit1_threshold_start=0.35, exit1_threshold_end=0.55,
                 exit2_threshold_start=0.45, exit2_threshold_end=0.65,
                 gumbel_temperature=1.0, num_epochs=20,
                 min_detections=2, fg_threshold=0.60, warmup_epochs=3):
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
        self.exit2_context_proj = nn.Conv2d(128, 112, kernel_size=1)
        self.exit2_branch = EnhancedEarlyExitBranch(
            in_channels=112,
            intermediate_channels=224,
            num_anchors=num_anchors,
            num_classes=2,
            use_attention=True
        )

        # Full branch: 672 channels
        self.full_context_proj = nn.Conv2d(224, 672, kernel_size=1)
        self.full_branch = EnhancedEarlyExitBranch(
            in_channels=672,
            intermediate_channels=512,
            num_anchors=num_anchors,
            num_classes=2,
            use_attention=True
        )

        # PHASE 1.9: MODERATE Curriculum Learning Thresholds
        self.exit1_threshold_start = exit1_threshold_start
        self.exit1_threshold_end = exit1_threshold_end
        self.exit2_threshold_start = exit2_threshold_start
        self.exit2_threshold_end = exit2_threshold_end
        self.num_epochs = num_epochs

        # PHASE 1.9: RELAXED Quality gate parameters
        self.min_detections = min_detections  # Reduced from 3 to 2
        self.fg_threshold = fg_threshold      # Reduced from 0.70 to 0.60

        # PHASE 1.9: Warmup period
        self.warmup_epochs = warmup_epochs

        # Gumbel-Softmax temperature
        self.gumbel_temperature = gumbel_temperature

        self.num_classes = 2

        # Temperature parameters with clamping (keep from Phase 1.8)
        self.temperature1 = nn.Parameter(torch.ones(1) * 1.5)
        self.temperature2 = nn.Parameter(torch.ones(1) * 1.5)
        self.temperature_full = nn.Parameter(torch.ones(1) * 1.0)

        # Temperature constraints
        self.temp_min = 0.5
        self.temp_max = 3.0

        # Statistics tracking
        self.exit_stats = {'exit1': 0, 'exit2': 0, 'full': 0}
        self.training_step = 0
        self.current_epoch = 0

    def get_curriculum_thresholds(self, epoch):
        """
        PHASE 1.9: MODERATE curriculum learning threshold schedule

        Start: 0.35, 0.45 (between Phase 1.7's 0.25/0.35 and Phase 1.8's 0.40/0.50)
        End: 0.55, 0.65 (between Phase 1.7's 0.45/0.60 and Phase 1.8's 0.65/0.75)
        """
        progress = min(epoch / self.num_epochs, 1.0)

        # Linear interpolation from start to end thresholds
        exit1_threshold = self.exit1_threshold_start + progress * (self.exit1_threshold_end - self.exit1_threshold_start)
        exit2_threshold = self.exit2_threshold_start + progress * (self.exit2_threshold_end - self.exit2_threshold_start)

        return exit1_threshold, exit2_threshold

    def get_warmup_weight(self, epoch):
        """
        PHASE 1.9: Warmup schedule for early exit training

        Epochs 0-2 (warmup_epochs=3): weight = 0 (full model only)
        Epochs 3-5: gradual increase 0.2 -> 0.6 -> 1.0
        Epochs 6+: full weight = 1.0
        """
        if epoch < self.warmup_epochs:
            return 0.0  # No early exit training during warmup
        elif epoch < self.warmup_epochs + 3:
            # Gradual increase: 0.2, 0.6, 1.0
            progress = (epoch - self.warmup_epochs) / 3.0
            return 0.2 + 0.8 * progress
        else:
            return 1.0  # Full training

    def compute_confidence(self, cls_logits, temperature, bbox_regression=None):
        """
        PHASE 1.9: SMOOTH confidence gating with sigmoid

        Instead of binary pass/fail (Phase 1.8), use sigmoid smoothing:
        - Gradually penalize insufficient detections
        - Still focuses on foreground class probability
        - More forgiving for early exits with lower capacity
        """
        batch_size = cls_logits.shape[0]

        # Reshape logits: [B, C, H, W] -> [B, num_anchors, C]
        cls_logits = cls_logits.permute(0, 2, 3, 1).contiguous()
        cls_logits = cls_logits.view(batch_size, -1, self.num_classes)

        # Clamp temperature to prevent explosion (keep from Phase 1.8)
        temperature_clamped = torch.clamp(temperature, self.temp_min, self.temp_max)

        # Apply temperature scaling
        scaled_logits = cls_logits / temperature_clamped

        # Apply softmax to get probabilities
        probs = F.softmax(scaled_logits, dim=-1)

        # Focus on FOREGROUND class (class 1 = chair)
        fg_probs = probs[:, :, 1]  # [batch_size, num_anchors]

        # Count confident foreground detections per sample
        confident_fg = (fg_probs > self.fg_threshold).float().sum(dim=1)  # [batch_size]

        # Average foreground probability
        avg_fg_prob = fg_probs.mean(dim=1)  # [batch_size]

        # PHASE 1.9: SMOOTH QUALITY GATE with sigmoid
        # Instead of hard cutoff, use smooth transition
        # sigmoid((x - threshold) * scale) gives smooth 0->1 transition
        detection_score = torch.sigmoid((confident_fg - self.min_detections) * 2.0)

        # Final confidence: weighted by detection quality (smooth, not binary)
        confidence = avg_fg_prob * detection_score

        return confidence

    def forward(self, images, targets=None, epoch=0):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be provided")

        # Transform images if needed
        if isinstance(images, (list, tuple)):
            images = torch.stack(images)

        # Get curriculum thresholds for current epoch
        curr_exit1_threshold, curr_exit2_threshold = self.get_curriculum_thresholds(epoch)

        # ============ Exit 1 (Layer 8) ============
        features_exit1 = self.exit1_features(images)
        exit1_cls, exit1_reg = self.exit1_branch(features_exit1)
        exit1_confidence = self.compute_confidence(exit1_cls, self.temperature1, exit1_reg)

        if self.training:
            # During training, compute all paths and use Gumbel-Softmax for exit decisions

            # Continue to exit 2
            features_mid = self.mid_features(features_exit1)

            # Exit 2 with context from exit 1
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
            exit2_confidence = self.compute_confidence(exit2_cls, self.temperature2, exit2_reg)

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
            full_confidence = self.compute_confidence(full_cls, self.temperature_full, full_reg)

            # Compute losses with Gumbel-Softmax exit decisions and warmup
            losses = self.compute_training_losses(
                exit1_cls, exit1_reg, exit1_confidence,
                exit2_cls, exit2_reg, exit2_confidence,
                full_cls, full_reg, full_confidence,
                targets, curr_exit1_threshold, curr_exit2_threshold, epoch
            )

            self.training_step += 1
            return losses

        else:
            # During inference - cascade evaluation with curriculum thresholds
            avg_confidence_exit1 = exit1_confidence.mean().item()

            if avg_confidence_exit1 >= curr_exit1_threshold:
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
            exit2_confidence = self.compute_confidence(exit2_cls, self.temperature2, exit2_reg)
            avg_confidence_exit2 = exit2_confidence.mean().item()

            if avg_confidence_exit2 >= curr_exit2_threshold:
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
            full_confidence = self.compute_confidence(full_cls, self.temperature_full, full_reg)

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
                                full_cls, full_reg, full_conf, targets,
                                curr_exit1_threshold, curr_exit2_threshold, epoch):
        """
        PHASE 1.9: Compute losses with warmup period and smooth gating
        """

        batch_size = full_cls.shape[0]
        num_predictions = full_cls.view(batch_size, -1, self.num_classes).shape[1]

        # Prepare ground truth targets for training
        cls_targets, reg_targets, reg_weights = self.prepare_targets_for_loss(full_cls, full_reg, targets)

        # PHASE 1.9: Get warmup weight
        warmup_weight = self.get_warmup_weight(epoch)

        # ========== Gumbel-Softmax Exit Decisions ==========
        # Create logits for exit decisions based on confidence vs threshold
        # Shape: [batch_size, 2] where [:, 0] = don't exit, [:, 1] = exit

        # Exit 1 decision logits
        exit1_decision_logits = torch.stack([
            -(exit1_conf - curr_exit1_threshold) * 10,  # Don't exit logit
            (exit1_conf - curr_exit1_threshold) * 10     # Exit logit
        ], dim=1)

        # Sample discrete decision using Gumbel-Softmax
        exit1_decision = gumbel_softmax_sample(exit1_decision_logits,
                                              temperature=self.gumbel_temperature,
                                              hard=True)  # [batch, 2]
        exit1_would_exit = exit1_decision[:, 1]  # Probability of exiting at exit1

        # Exit 2 decision logits (only for samples that didn't exit at exit1)
        exit2_decision_logits = torch.stack([
            -(exit2_conf - curr_exit2_threshold) * 10,
            (exit2_conf - curr_exit2_threshold) * 10
        ], dim=1)

        exit2_decision = gumbel_softmax_sample(exit2_decision_logits,
                                              temperature=self.gumbel_temperature,
                                              hard=True)
        exit2_would_exit = (1 - exit1_would_exit) * exit2_decision[:, 1]

        # Calculate actual exit rates from hard decisions
        actual_exit1_rate = exit1_would_exit.float().mean()
        actual_exit2_rate = exit2_would_exit.float().mean()

        # ========== Full model loss (supervised with REAL targets) ==========
        full_cls_loss = F.cross_entropy(
            full_cls.view(batch_size, num_predictions, self.num_classes).reshape(-1, self.num_classes),
            cls_targets.view(-1)
        )

        # Regression loss with ground truth boxes
        full_reg_flat = full_reg.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)
        if reg_weights.sum() > 0:
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

        exit2_cls_supervised = F.cross_entropy(
            exit2_cls_resized.view(batch_size, num_predictions, self.num_classes).reshape(-1, self.num_classes),
            cls_targets.view(-1)
        )

        exit2_cls_distill = F.kl_div(
            F.log_softmax(exit2_cls_resized.view(-1, self.num_classes), dim=1),
            full_cls_probs,
            reduction='batchmean'
        )

        exit2_cls_loss = 0.7 * exit2_cls_supervised + 0.3 * exit2_cls_distill

        exit2_reg_flat = exit2_reg_resized.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)
        if reg_weights.sum() > 0:
            exit2_reg_supervised = F.smooth_l1_loss(
                exit2_reg_flat * reg_weights.unsqueeze(-1),
                reg_targets * reg_weights.unsqueeze(-1),
                reduction='sum'
            ) / reg_weights.sum()
        else:
            exit2_reg_supervised = F.smooth_l1_loss(exit2_reg_flat, torch.zeros_like(exit2_reg_flat)) * 0.01

        exit2_reg_distill = F.smooth_l1_loss(exit2_reg_resized, full_reg.detach())
        exit2_reg_loss = 0.7 * exit2_reg_supervised + 0.3 * exit2_reg_distill

        exit2_loss = exit2_cls_loss + exit2_reg_loss

        # ========== Exit 1 loss (supervised + distillation from exit 2) ==========
        with torch.no_grad():
            exit2_cls_probs = F.softmax(exit2_cls.view(-1, self.num_classes), dim=1)

        if exit1_cls.shape != exit2_cls.shape:
            exit1_cls_resized = F.interpolate(exit1_cls, size=exit2_cls.shape[2:], mode='bilinear', align_corners=False)
            exit1_reg_resized = F.interpolate(exit1_reg, size=exit2_reg.shape[2:], mode='bilinear', align_corners=False)
        else:
            exit1_cls_resized = exit1_cls
            exit1_reg_resized = exit1_reg

        exit1_num_preds = exit1_cls_resized.view(batch_size, -1, self.num_classes).shape[1]
        if exit1_num_preds != num_predictions:
            exit1_cls_targets = cls_targets[:, :exit1_num_preds] if exit1_num_preds < num_predictions else cls_targets
        else:
            exit1_cls_targets = cls_targets

        exit1_cls_supervised = F.cross_entropy(
            exit1_cls_resized.view(batch_size, -1, self.num_classes).reshape(-1, self.num_classes),
            exit1_cls_targets.view(-1)
        )

        exit1_cls_distill = F.kl_div(
            F.log_softmax(exit1_cls_resized.view(-1, self.num_classes), dim=1),
            exit2_cls_probs,
            reduction='batchmean'
        )

        exit1_cls_loss = 0.7 * exit1_cls_supervised + 0.3 * exit1_cls_distill

        exit1_reg_flat = exit1_reg_resized.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)
        if reg_weights.sum() > 0:
            exit1_reg_supervised = F.smooth_l1_loss(
                exit1_reg_flat * reg_weights.unsqueeze(-1),
                reg_targets * reg_weights.unsqueeze(-1),
                reduction='sum'
            ) / reg_weights.sum()
        else:
            exit1_reg_supervised = F.smooth_l1_loss(exit1_reg_flat, torch.zeros_like(exit1_reg_flat)) * 0.01

        exit1_reg_distill = F.smooth_l1_loss(exit1_reg_resized, exit2_reg.detach())
        exit1_reg_loss = 0.7 * exit1_reg_supervised + 0.3 * exit1_reg_distill

        exit1_loss = exit1_cls_loss + exit1_reg_loss

        # ========== Cascade consistency loss ==========
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
        conf_diversity_loss = torch.tensor(0.0, device=full_cls.device)

        if exit1_conf.mean() > exit2_conf.mean():
            conf_diversity_loss += (exit1_conf.mean() - exit2_conf.mean())

        if exit2_conf.mean() > full_conf.mean():
            conf_diversity_loss += (exit2_conf.mean() - full_conf.mean())

        # ========== Distribution loss with HARD exits from Gumbel-Softmax ==========
        # Target: 30% Exit1, 25% Exit2, 45% Full
        target_exit1_rate = 0.30
        target_exit2_rate = 0.25

        # Using ACTUAL hard exit decisions
        distribution_loss = (
            (actual_exit1_rate - target_exit1_rate) ** 2 +
            (actual_exit2_rate - target_exit2_rate) ** 2
        )

        # ========== PHASE 1.9: Loss weighting with WARMUP ==========
        epoch_progress = min(self.training_step / 5000.0, 1.0)

        # Start: (0.2, 0.3, 0.5) -> End: (0.3, 0.4, 0.3)
        alpha_exit1 = (0.2 + 0.1 * epoch_progress) * warmup_weight  # Scale by warmup
        alpha_exit2 = (0.3 + 0.1 * epoch_progress) * warmup_weight  # Scale by warmup
        alpha_full = 0.5 - 0.2 * epoch_progress * warmup_weight     # Always train full, but reduce

        # Ensure full model gets trained even during warmup
        alpha_full = max(alpha_full, 0.5)

        # Total loss with distribution guidance and warmup
        total_loss = (
            alpha_exit1 * exit1_loss +
            alpha_exit2 * exit2_loss +
            alpha_full * full_loss +
            0.05 * cascade_loss * warmup_weight +
            0.02 * conf_diversity_loss * warmup_weight +
            0.1 * distribution_loss * warmup_weight
        )

        # Log clamped temperatures
        temp1_clamped = torch.clamp(self.temperature1, self.temp_min, self.temp_max)
        temp2_clamped = torch.clamp(self.temperature2, self.temp_min, self.temp_max)
        temp_full_clamped = torch.clamp(self.temperature_full, self.temp_min, self.temp_max)

        return {
            'total_loss': total_loss,
            'exit1_loss': exit1_loss,
            'exit2_loss': exit2_loss,
            'full_loss': full_loss,
            'cascade_loss': cascade_loss,
            'diversity_loss': conf_diversity_loss,
            'distribution_loss': distribution_loss,
            'actual_exit1_rate': actual_exit1_rate,
            'actual_exit2_rate': actual_exit2_rate,
            'exit1_confidence': exit1_conf.mean(),
            'exit2_confidence': exit2_conf.mean(),
            'full_confidence': full_conf.mean(),
            'confidence_std_exit1': exit1_conf.std(),
            'confidence_std_exit2': exit2_conf.std(),
            'curriculum_exit1_threshold': torch.tensor(curr_exit1_threshold),
            'curriculum_exit2_threshold': torch.tensor(curr_exit2_threshold),
            'temperature1_clamped': temp1_clamped.item(),
            'temperature2_clamped': temp2_clamped.item(),
            'temperature_full_clamped': temp_full_clamped.item(),
            'warmup_weight': warmup_weight,
        }


class Trainer:
    """Training class for multi-level early exit model"""

    def __init__(self, model, train_loader, val_loader, device, num_epochs):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs

        # Optimizer with different learning rates
        self.optimizer = torch.optim.AdamW([
            {'params': model.exit1_branch.parameters(), 'lr': 1e-3},
            {'params': model.exit2_branch.parameters(), 'lr': 1e-3},
            {'params': model.full_branch.parameters(), 'lr': 1e-3},
            {'params': model.backbone.parameters(), 'lr': 1e-4},
            {'params': [model.temperature1, model.temperature2, model.temperature_full], 'lr': 1e-2},
        ], weight_decay=0.0005)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs
        )

        # Metrics tracking
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        self.model.current_epoch = epoch

        epoch_metrics = defaultdict(list)

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}')
        for batch_idx, (images, targets) in enumerate(pbar):
            try:
                # Move to device
                images = torch.stack([img.to(self.device) for img in images])
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                # Forward pass with current epoch for curriculum
                losses = self.model(images, targets, epoch=epoch)

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
                    elif isinstance(value, (int, float)):
                        epoch_metrics[key].append(value)

                # Update progress bar
                if batch_idx % 10 == 0:
                    curr_th1, curr_th2 = self.model.get_curriculum_thresholds(epoch)
                    warmup_w = self.model.get_warmup_weight(epoch)
                    pbar.set_postfix({
                        'loss': np.mean(epoch_metrics['total_loss'][-50:]),
                        'e1_conf': np.mean(epoch_metrics['exit1_confidence'][-50:]),
                        'e2_conf': np.mean(epoch_metrics['exit2_confidence'][-50:]),
                        'warmup': f'{warmup_w:.2f}',
                        'curr_th1': f'{curr_th1:.3f}',
                        'e1_rate': np.mean(epoch_metrics['actual_exit1_rate'][-50:])
                    })

            except RuntimeError as e:
                print(f"Error at batch {batch_idx}: {e}")
                continue

        # Store epoch metrics
        for key, values in epoch_metrics.items():
            self.train_metrics[key].append(np.mean(values))

    def validate(self, epoch):
        """Validate model"""
        self.model.eval()
        self.model.current_epoch = epoch

        inference_times = []
        confidences_per_exit = {'exit1': [], 'exit2': [], 'full': []}

        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc='Validation'):
                images = torch.stack([img.to(self.device) for img in images])

                # Measure inference time
                start_time = time.time()
                outputs = self.model(images, epoch=epoch)
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

    def train(self):
        """Full training loop"""
        for epoch in range(self.num_epochs):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            print(f"{'='*70}")

            # Get curriculum thresholds and warmup weight
            curr_th1, curr_th2 = self.model.get_curriculum_thresholds(epoch)
            warmup_w = self.model.get_warmup_weight(epoch)

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
            print(f"  Distribution Loss: {self.train_metrics['distribution_loss'][-1]:.4f}")

            print(f"\n--- PHASE 1.9: Warmup Status (Epoch {epoch + 1}) ---")
            print(f"  Warmup Weight: {warmup_w:.2f} (0=warmup, 1=full training)")
            if epoch < self.model.warmup_epochs:
                print(f"  🔥 WARMUP MODE: Training full model only (epochs 1-{self.model.warmup_epochs})")
            elif epoch < self.model.warmup_epochs + 3:
                print(f"  ⬆️ RAMP-UP MODE: Gradually enabling early exits")
            else:
                print(f"  ✅ FULL MODE: All branches training")

            print(f"\n--- Confidence Progression ---")
            print(f"  Exit 1: {self.train_metrics['exit1_confidence'][-1]:.3f} (±{self.train_metrics['confidence_std_exit1'][-1]:.3f})")
            print(f"  Exit 2: {self.train_metrics['exit2_confidence'][-1]:.3f} (±{self.train_metrics['confidence_std_exit2'][-1]:.3f})")
            print(f"  Full:   {self.train_metrics['full_confidence'][-1]:.3f}")

            print(f"\n--- PHASE 1.9: MODERATE Curriculum Thresholds (Epoch {epoch + 1}) ---")
            print(f"  Exit 1 Threshold: {curr_th1:.3f} (conf: {self.train_metrics['exit1_confidence'][-1]:.3f})")
            print(f"  Exit 2 Threshold: {curr_th2:.3f} (conf: {self.train_metrics['exit2_confidence'][-1]:.3f})")
            print(f"  Progress: {(epoch / self.num_epochs * 100):.1f}%  |  Start: (0.35, 0.45) -> End: (0.55, 0.65)")

            print(f"\n--- PHASE 1.9: Actual Exit Distribution (Gumbel-Softmax) ---")
            print(f"  Training Exit 1 Rate: {self.train_metrics['actual_exit1_rate'][-1]:.2%} (target: 30%)")
            print(f"  Training Exit 2 Rate: {self.train_metrics['actual_exit2_rate'][-1]:.2%} (target: 25%)")
            print(f"  Training Full Rate:   {1 - self.train_metrics['actual_exit1_rate'][-1] - self.train_metrics['actual_exit2_rate'][-1]:.2%} (target: 45%)")

            print(f"\n--- PHASE 1.9: Clamped Temperature Values ---")
            print(f"  Exit 1: {self.train_metrics['temperature1_clamped'][-1]:.3f} (constrained: 0.5-3.0)")
            print(f"  Exit 2: {self.train_metrics['temperature2_clamped'][-1]:.3f} (constrained: 0.5-3.0)")
            print(f"  Full:   {self.train_metrics['temperature_full_clamped'][-1]:.3f} (constrained: 0.5-3.0)")

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
        final_model_path = os.path.join(MODEL_OUTPUT_PATH, 'early_exit_ssdlite_phase1_9_final.pth')
        torch.save(self.model.state_dict(), final_model_path)
        print(f"\nFinal model saved: {final_model_path}")

        # Save metrics to CSV
        train_df = pd.DataFrame(self.train_metrics)
        train_df.to_csv(os.path.join(ANALYSIS_EE_OUTPUT_PATH, 'train_metrics_phase1_9.csv'), index=False)

        val_df = pd.DataFrame(self.val_metrics)
        val_df.to_csv(os.path.join(ANALYSIS_EE_OUTPUT_PATH, 'val_metrics_phase1_9.csv'), index=False)

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

        # Row 1, Col 2: Confidence progression with curriculum thresholds
        axes[0, 1].plot(epochs, self.train_metrics['exit1_confidence'], 'r-', label='Exit 1 Conf', linewidth=2)
        axes[0, 1].plot(epochs, self.train_metrics['exit2_confidence'], 'g-', label='Exit 2 Conf', linewidth=2)
        axes[0, 1].plot(epochs, self.train_metrics['full_confidence'], 'm-', label='Full Conf', linewidth=2)
        axes[0, 1].plot(epochs, self.train_metrics['curriculum_exit1_threshold'], 'r:', label='Curr Thresh 1', linewidth=2, alpha=0.7)
        axes[0, 1].plot(epochs, self.train_metrics['curriculum_exit2_threshold'], 'g:', label='Curr Thresh 2', linewidth=2, alpha=0.7)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Confidence / Threshold')
        axes[0, 1].set_title('PHASE 1.9: Confidence vs MODERATE Curriculum')
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

        # Row 2, Col 1: Warmup weight progression
        axes[1, 0].plot(epochs, self.train_metrics['warmup_weight'], 'purple', linewidth=2)
        axes[1, 0].axvline(x=self.model.warmup_epochs, color='red', linestyle='--', alpha=0.5, label=f'Warmup End (epoch {self.model.warmup_epochs})')
        axes[1, 0].axvline(x=self.model.warmup_epochs + 3, color='orange', linestyle='--', alpha=0.5, label=f'Ramp-up End (epoch {self.model.warmup_epochs + 3})')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Warmup Weight')
        axes[1, 0].set_title('PHASE 1.9: Warmup Schedule')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Row 2, Col 2: Inference time
        axes[1, 1].plot(epochs, self.val_metrics['inference_time'], 'orange', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Time (ms)')
        axes[1, 1].set_title('Average Inference Time')
        axes[1, 1].grid(True, alpha=0.3)

        # Row 2, Col 3: Training vs Validation exit rates
        axes[1, 2].plot(epochs, self.train_metrics['actual_exit1_rate'], 'r--', label='Train Exit 1', linewidth=2, alpha=0.7)
        axes[1, 2].plot(epochs, self.train_metrics['actual_exit2_rate'], 'g--', label='Train Exit 2', linewidth=2, alpha=0.7)
        axes[1, 2].plot(epochs, self.val_metrics['exit1_rate'], 'r-', label='Val Exit 1', linewidth=2)
        axes[1, 2].plot(epochs, self.val_metrics['exit2_rate'], 'g-', label='Val Exit 2', linewidth=2)
        axes[1, 2].axhline(y=0.30, color='red', linestyle=':', alpha=0.5, label='Target 30%')
        axes[1, 2].axhline(y=0.25, color='green', linestyle=':', alpha=0.5, label='Target 25%')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Exit Rate')
        axes[1, 2].set_title('PHASE 1.9: Training-Validation Alignment')
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

        # Row 3, Col 2: Temperature evolution
        axes[2, 1].plot(epochs, self.train_metrics['temperature1_clamped'], 'r-', label='Exit 1 Temp', linewidth=2)
        axes[2, 1].plot(epochs, self.train_metrics['temperature2_clamped'], 'g-', label='Exit 2 Temp', linewidth=2)
        axes[2, 1].plot(epochs, self.train_metrics['temperature_full_clamped'], 'm-', label='Full Temp', linewidth=2)
        axes[2, 1].axhline(y=self.model.temp_min, color='k', linestyle='--', alpha=0.3, label=f'Min: {self.model.temp_min}')
        axes[2, 1].axhline(y=self.model.temp_max, color='k', linestyle=':', alpha=0.3, label=f'Max: {self.model.temp_max}')
        axes[2, 1].set_xlabel('Epoch')
        axes[2, 1].set_ylabel('Temperature Value')
        axes[2, 1].set_title('PHASE 1.9: Clamped Temperature Evolution')
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
        plt.savefig(os.path.join(ANALYSIS_EE_OUTPUT_PATH, 'training_plots_phase1_9.png'), dpi=150, bbox_inches='tight')
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
    """Main training function for Phase 1.9"""
    # Configuration
    BATCH_SIZE = 4
    NUM_EPOCHS = 20

    # PHASE 1.9: MODERATE Curriculum Learning Thresholds
    EXIT1_THRESHOLD_START = 0.35  # Between Phase 1.7's 0.25 and Phase 1.8's 0.40
    EXIT1_THRESHOLD_END = 0.55    # Between Phase 1.7's 0.45 and Phase 1.8's 0.65
    EXIT2_THRESHOLD_START = 0.45  # Between Phase 1.7's 0.35 and Phase 1.8's 0.50
    EXIT2_THRESHOLD_END = 0.65    # Between Phase 1.7's 0.60 and Phase 1.8's 0.75

    # Gumbel-Softmax temperature
    GUMBEL_TEMPERATURE = 0.5

    # PHASE 1.9: RELAXED quality gate parameters
    MIN_DETECTIONS = 2     # Reduced from 3 (Phase 1.8)
    FG_THRESHOLD = 0.60    # Reduced from 0.70 (Phase 1.8)

    # PHASE 1.9: Warmup period
    WARMUP_EPOCHS = 3      # First 3 epochs: full model only

    NUM_WORKERS = 0

    print("="*80)
    print(" Phase 1.9: BALANCED Approach - Moderate + Relaxed + Warmup")
    print("="*80)
    print()
    print("PHASE 1.8 FAILURE ANALYSIS:")
    print("  ❌ Exit1/Exit2 confidence collapsed to 0.000")
    print("  ❌ 0% early exits (100% Full model - defeats purpose)")
    print("  ❌ Quality gate too strict (MIN_DET=3, FG_THRESH=0.70)")
    print("  ❌ Curriculum too high (0.40→0.65, 0.50→0.75)")
    print("  ✅ Temperature clamping WORKS (1.55, 2.15 within 0.5-3.0)")
    print("="*80)
    print()
    print("PHASE 1.9 SOLUTIONS - GOLDILOCKS ZONE:")
    print("  ✅ FIX 1: MODERATE CURRICULUM (not too high, not too low)")
    print(f"     - Exit 1: {EXIT1_THRESHOLD_START:.2f} → {EXIT1_THRESHOLD_END:.2f}")
    print(f"     - Exit 2: {EXIT2_THRESHOLD_START:.2f} → {EXIT2_THRESHOLD_END:.2f}")
    print("     - Between Phase 1.7's too-low and Phase 1.8's too-high")
    print()
    print("  ✅ FIX 2: RELAXED QUALITY GATES")
    print(f"     - MIN_DETECTIONS: {MIN_DETECTIONS} (down from 3)")
    print(f"     - FG_THRESHOLD: {FG_THRESHOLD} (down from 0.70)")
    print("     - Early layers have less capacity, shouldn't need full-model standards")
    print()
    print("  ✅ FIX 3: WARMUP PERIOD")
    print(f"     - Epochs 1-{WARMUP_EPOCHS}: Full model only (no early exit losses)")
    print(f"     - Epochs {WARMUP_EPOCHS+1}-{WARMUP_EPOCHS+3}: Gradual ramp-up (0.2x → 0.6x → 1.0x)")
    print(f"     - Epochs {WARMUP_EPOCHS+4}+: Full cascade training")
    print("     - Let early exits learn features before requiring exits")
    print()
    print("  ✅ FIX 4: SMOOTH CONFIDENCE GATING")
    print("     - Sigmoid smoothing instead of binary pass/fail")
    print("     - Gradual penalty instead of hard rejection")
    print()
    print("  ✅ KEEP WHAT WORKS FROM PHASE 1.8:")
    print("     - Temperature clamping (0.5-3.0)")
    print("     - Foreground-aware confidence metric")
    print("     - Cascade architecture")
    print("="*80)
    print()

    # Create output directories
    create_output_directories()
    print()

    print(f"Configuration:")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Exit 1 Threshold: {EXIT1_THRESHOLD_START:.2f} -> {EXIT1_THRESHOLD_END:.2f} (MODERATE curriculum)")
    print(f"  Exit 2 Threshold: {EXIT2_THRESHOLD_START:.2f} -> {EXIT2_THRESHOLD_END:.2f} (MODERATE curriculum)")
    print(f"  Gumbel Temperature: {GUMBEL_TEMPERATURE}")
    print(f"  Temperature Constraints: [0.5, 3.0]")
    print(f"  Min Detections: {MIN_DETECTIONS} (RELAXED)")
    print(f"  Foreground Threshold: {FG_THRESHOLD} (RELAXED)")
    print(f"  Warmup Epochs: {WARMUP_EPOCHS}")
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
    print("Initializing Phase 1.9 model with BALANCED parameters...")
    from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights
    base_model = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
    model = MultiLevelCascadeEarlyExitSSDLite(
        base_model,
        exit1_threshold_start=EXIT1_THRESHOLD_START,
        exit1_threshold_end=EXIT1_THRESHOLD_END,
        exit2_threshold_start=EXIT2_THRESHOLD_START,
        exit2_threshold_end=EXIT2_THRESHOLD_END,
        gumbel_temperature=GUMBEL_TEMPERATURE,
        num_epochs=NUM_EPOCHS,
        min_detections=MIN_DETECTIONS,
        fg_threshold=FG_THRESHOLD,
        warmup_epochs=WARMUP_EPOCHS
    )

    print("Architecture Details:")
    print(f"  Exit 1: Layer 8 (80->128 channels) + SE Attention")
    print(f"  Exit 2: Layer 12 (112->224 channels) + SE Attention")
    print(f"  Full:   Complete backbone (672 channels)")
    print(f"  Cascade: Each level refines previous predictions")
    print(f"  ** PHASE 1.9: BALANCED strictness + Warmup + Smooth gating")
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
    trainer = Trainer(model, train_loader, val_loader, device, NUM_EPOCHS)

    # Train
    trainer.train()

    print("\n" + "="*80)
    print(" Training Complete - Phase 1.9")
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
        'final_distribution_loss': trainer.train_metrics['distribution_loss'][-1],

        'final_exit1_rate': trainer.val_metrics['exit1_rate'][-1],
        'final_exit2_rate': trainer.val_metrics['exit2_rate'][-1],
        'final_full_rate': trainer.val_metrics['full_rate'][-1],
        'final_combined_early_exit_rate': trainer.val_metrics['combined_early_exit_rate'][-1],

        'final_inference_time': trainer.val_metrics['inference_time'][-1],

        'train_exit1_confidence': trainer.train_metrics['exit1_confidence'][-1],
        'train_exit2_confidence': trainer.train_metrics['exit2_confidence'][-1],
        'train_full_confidence': trainer.train_metrics['full_confidence'][-1],

        'actual_exit1_rate': trainer.train_metrics['actual_exit1_rate'][-1],
        'actual_exit2_rate': trainer.train_metrics['actual_exit2_rate'][-1],

        'val_exit1_confidence': trainer.val_metrics['exit1_confidence'][-1],
        'val_exit2_confidence': trainer.val_metrics['exit2_confidence'][-1],
        'val_full_confidence': trainer.val_metrics['full_confidence'][-1],

        'exit1_threshold_start': EXIT1_THRESHOLD_START,
        'exit1_threshold_end': EXIT1_THRESHOLD_END,
        'exit2_threshold_start': EXIT2_THRESHOLD_START,
        'exit2_threshold_end': EXIT2_THRESHOLD_END,
        'gumbel_temperature': GUMBEL_TEMPERATURE,
        'min_detections': MIN_DETECTIONS,
        'fg_threshold': FG_THRESHOLD,
        'warmup_epochs': WARMUP_EPOCHS,

        'temperature1_final': trainer.train_metrics['temperature1_clamped'][-1],
        'temperature2_final': trainer.train_metrics['temperature2_clamped'][-1],
        'temperature_full_final': trainer.train_metrics['temperature_full_clamped'][-1],

        'exit1_count': trainer.val_metrics['exit1_count'],
        'exit2_count': trainer.val_metrics['exit2_count'],
        'full_count': trainer.val_metrics['full_count']
    }

    # Save final summary
    with open(os.path.join(ANALYSIS_EE_OUTPUT_PATH, 'final_summary_phase1_9.json'), 'w') as f:
        json.dump(final_metrics, f, indent=4)

    # Print comprehensive summary
    print("="*80)
    print(" PHASE 1.9 FINAL RESULTS")
    print("="*80)
    print()

    print("--- Training Losses ---")
    print(f"  Total:        {final_metrics['final_train_loss']:.4f}")
    print(f"  Exit 1:       {final_metrics['final_exit1_loss']:.4f}")
    print(f"  Exit 2:       {final_metrics['final_exit2_loss']:.4f}")
    print(f"  Full:         {final_metrics['final_full_loss']:.4f}")
    print(f"  Cascade:      {final_metrics['final_cascade_loss']:.4f}")
    print(f"  Distribution: {final_metrics['final_distribution_loss']:.4f}")
    print()

    print("--- Exit Distribution (Validation) ---")
    print(f"  Exit 1:   {final_metrics['final_exit1_rate']:.2%} ({final_metrics['exit1_count']} samples)")
    print(f"  Exit 2:   {final_metrics['final_exit2_rate']:.2%} ({final_metrics['exit2_count']} samples)")
    print(f"  Full:     {final_metrics['final_full_rate']:.2%} ({final_metrics['full_count']} samples)")
    print(f"  Combined Early Exit: {final_metrics['final_combined_early_exit_rate']:.2%}")
    print()

    print("--- Confidence Values ---")
    print(f"  Training:")
    print(f"    Exit 1:  {final_metrics['train_exit1_confidence']:.3f}")
    print(f"    Exit 2:  {final_metrics['train_exit2_confidence']:.3f}")
    print(f"    Full:    {final_metrics['train_full_confidence']:.3f}")
    print(f"  Validation:")
    print(f"    Exit 1:  {final_metrics['val_exit1_confidence']:.3f}")
    print(f"    Exit 2:  {final_metrics['val_exit2_confidence']:.3f}")
    print(f"    Full:    {final_metrics['val_full_confidence']:.3f}")
    print()

    print("--- PHASE 1.9: Temperature Control (Clamped) ---")
    print(f"  Exit 1: {final_metrics['temperature1_final']:.3f} (range: 0.5-3.0)")
    print(f"  Exit 2: {final_metrics['temperature2_final']:.3f} (range: 0.5-3.0)")
    print(f"  Full:   {final_metrics['temperature_full_final']:.3f} (range: 0.5-3.0)")
    print()

    print("--- Performance ---")
    print(f"  Avg Inference Time: {final_metrics['final_inference_time']:.2f}ms")
    print()

    print("="*80)
    print(" COMPARISON: Phase 1.7 vs Phase 1.8 vs Phase 1.9")
    print("="*80)
    print("  Phase 1.7: 100% Exit1 (confidence=0.500, temp=24.56) - TOO PERMISSIVE")
    print("  Phase 1.8: 0% early exit (confidence=0.000) - TOO STRICT")
    print(f"  Phase 1.9: {final_metrics['final_combined_early_exit_rate']:.2%} combined early exit - BALANCED?")
    print()
    print("  KEY IMPROVEMENTS:")
    print("    ✅ Moderate curriculum: 0.35→0.55, 0.45→0.65 (Goldilocks zone)")
    print("    ✅ Relaxed quality gates: min_det=2, fg_thresh=0.60")
    print("    ✅ Warmup period: 3 epochs full-only, then gradual ramp-up")
    print("    ✅ Smooth confidence gating: sigmoid instead of binary")
    print("="*80)


if __name__ == "__main__":
    main()
