# early_exit_training_system_v2_fixed.py

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


class EarlyExitBranch(nn.Module):
    """Early exit branch for SSDLite"""
    
    def __init__(self, in_channels, num_anchors, num_classes=2):
        super().__init__()
        
        # Lightweight detection head similar to SSDLite structure
        intermediate_channels = max(in_channels // 2, 64)
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, intermediate_channels, kernel_size=1),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, 
                     padding=1, groups=intermediate_channels),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=1),
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
    
    def forward(self, x):
        features = self.feature_extractor(x)
        cls_logits = self.cls_head(features)
        bbox_regression = self.reg_head(features)
        return cls_logits, bbox_regression


class EarlyExitSSDLite(nn.Module):
    """SSDLite with early exit capability"""

    def __init__(self, base_model, exit_threshold=0.7):
        super().__init__()

        # Store the complete original backbone
        self.backbone = base_model.backbone

        # Access first Sequential block for early exit point
        backbone_features = base_model.backbone.features
        first_sequential = list(backbone_features.children())[0]
        first_block_layers = list(first_sequential.children())

        # Early exit point after sublayer 7 (80 channels at 20x20 resolution)
        self.early_features = nn.Sequential(*first_block_layers[:8])

        # Get the number of channels at the early exit point
        early_channels = 80

        # Early exit branch
        num_anchors = 6  # Number of anchors per location
        self.early_branch = EarlyExitBranch(
            in_channels=early_channels,
            num_anchors=num_anchors,
            num_classes=2  # background + chair
        )

        # Full model branch (uses complete backbone output - 672 channels)
        full_channels = 672
        self.full_branch = EarlyExitBranch(
            in_channels=full_channels,
            num_anchors=num_anchors,
            num_classes=2  # background + chair
        )

        self.exit_threshold = exit_threshold
        self.num_classes = 2

        # Temperature parameter for confidence calibration
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

        # Statistics tracking
        self.exit_stats = {'early': 0, 'full': 0}
        self.training_step = 0

    def compute_confidence(self, cls_logits):
        """Compute confidence score using entropy-based approach with temperature scaling"""
        batch_size = cls_logits.shape[0]

        # Reshape logits
        cls_logits = cls_logits.permute(0, 2, 3, 1).contiguous()
        cls_logits = cls_logits.view(batch_size, -1, self.num_classes)

        # Apply temperature scaling for better calibration
        scaled_logits = cls_logits / self.temperature

        # Apply softmax to get probabilities
        probs = F.softmax(scaled_logits, dim=-1)

        # Compute entropy for each prediction
        # Higher entropy = more uncertainty = lower confidence
        epsilon = 1e-10  # For numerical stability
        entropy = -torch.sum(probs * torch.log(probs + epsilon), dim=-1)

        # Normalize entropy to [0, 1] range
        max_entropy = np.log(self.num_classes)  # Maximum possible entropy
        normalized_entropy = entropy / max_entropy

        # Convert entropy to confidence (1 - normalized_entropy)
        # Low entropy (certain) = high confidence
        # High entropy (uncertain) = low confidence
        confidence_per_prediction = 1.0 - normalized_entropy

        # Take mean confidence across all predictions for each sample
        # This gives a more stable measure than just using max
        mean_confidence = confidence_per_prediction.mean(dim=1)

        return mean_confidence
    
    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be provided")

        # Transform images if needed
        if isinstance(images, (list, tuple)):
            images = torch.stack(images)

        # Early exit branch - process early features
        features_early = self.early_features(images)
        early_cls, early_reg = self.early_branch(features_early)
        early_confidence = self.compute_confidence(early_cls)

        if self.training:
            # During training, always compute both paths
            # Full model uses complete backbone
            features_full = self.backbone(images)
            # Get features from first output (higher resolution feature map)
            if isinstance(features_full, dict):
                features_full = features_full['0']
            elif isinstance(features_full, (list, tuple)):
                features_full = features_full[0]

            # Get full model predictions
            full_cls, full_reg = self.full_branch(features_full)

            # Compute losses using knowledge distillation approach
            losses = {}

            # For early branch: use distillation from full model
            with torch.no_grad():
                full_cls_probs = F.softmax(full_cls.view(-1, self.num_classes), dim=1)

            # Early classification loss (distillation from full model)
            early_cls_loss = F.kl_div(
                F.log_softmax(early_cls.view(-1, self.num_classes), dim=1),
                full_cls_probs,
                reduction='batchmean'
            )

            # Early regression loss (match full model predictions)
            early_reg_loss = F.smooth_l1_loss(early_reg, full_reg.detach())

            # Full model uses simple supervised losses
            batch_size = full_cls.shape[0]
            num_predictions = full_cls.view(batch_size, -1, self.num_classes).shape[1]

            # Create background targets (most predictions should be background)
            cls_targets = torch.zeros(batch_size, num_predictions, dtype=torch.long, device=early_cls.device)

            full_cls_loss = F.cross_entropy(
                full_cls.view(batch_size, num_predictions, self.num_classes).reshape(-1, self.num_classes),
                cls_targets.view(-1)
            )

            # Regression loss with smaller weight
            full_reg_loss = F.smooth_l1_loss(full_reg, torch.zeros_like(full_reg)) * 0.1

            # Compute confidence diversity regularization
            # Encourage variety in confidence scores (avoid all high or all low)
            confidence_std = early_confidence.std()
            target_std = 0.15  # Target standard deviation for confidence
            diversity_loss = F.mse_loss(confidence_std, torch.tensor(target_std, device=early_confidence.device))

            # Add penalty for early exits that differ from full model
            # This encourages the early branch to only exit when it matches full model
            early_full_agreement = F.kl_div(
                F.log_softmax(early_cls.view(-1, self.num_classes), dim=1),
                F.softmax(full_cls.view(-1, self.num_classes).detach(), dim=1),
                reduction='batchmean'
            )

            # Improved loss weighting strategy
            # Start with lower alpha (more full model) and increase over training
            epoch_progress = min(self.training_step / 5000.0, 1.0)  # Gradually increase over 5000 steps

            # Base alpha starts at 0.2 (favor full model) and can go up to 0.5
            # But only if confidence is genuinely high
            confidence_mean = early_confidence.mean().item()
            base_alpha = 0.2 + 0.3 * epoch_progress

            # Only increase alpha if confidence is above threshold
            if confidence_mean >= self.exit_threshold:
                alpha = min(base_alpha + 0.2, 0.6)  # Max 60% weight on early branch
            else:
                alpha = base_alpha  # Keep low weight if confidence is low

            losses['early_loss'] = early_cls_loss + early_reg_loss
            losses['full_loss'] = full_cls_loss + full_reg_loss
            losses['diversity_loss'] = diversity_loss * 0.1  # Small weight for diversity
            losses['agreement_loss'] = early_full_agreement * 0.05  # Small penalty for disagreement

            # Total loss with all components
            losses['total_loss'] = (
                alpha * losses['early_loss'] +
                (1 - alpha) * losses['full_loss'] +
                losses['diversity_loss'] +
                losses['agreement_loss']
            )
            losses['early_confidence'] = early_confidence.mean()
            losses['confidence_std'] = confidence_std

            self.training_step += 1
            return losses

        else:
            # During inference
            avg_confidence = early_confidence.mean().item()

            if avg_confidence >= self.exit_threshold:
                self.exit_stats['early'] += 1
                # Return early predictions
                return {
                    'cls_logits': early_cls,
                    'bbox_regression': early_reg,
                    'exit_point': 'early',
                    'confidence': avg_confidence
                }
            else:
                self.exit_stats['full'] += 1
                # Continue with full model
                features_full = self.backbone(images)
                if isinstance(features_full, dict):
                    features_full = features_full['0']
                elif isinstance(features_full, (list, tuple)):
                    features_full = features_full[0]

                full_cls, full_reg = self.full_branch(features_full)

                return {
                    'cls_logits': full_cls,
                    'bbox_regression': full_reg,
                    'exit_point': 'full',
                    'confidence': avg_confidence
                }


class Trainer:
    """Training class for early exit model"""
    
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Optimizer with different learning rates for different components
        # Note: early_features is part of backbone, so we only include backbone once
        self.optimizer = torch.optim.AdamW([
            {'params': model.early_branch.parameters(), 'lr': 1e-3},
            {'params': model.full_branch.parameters(), 'lr': 1e-3},
            {'params': model.backbone.parameters(), 'lr': 1e-4},
            {'params': [model.temperature], 'lr': 1e-3},  # Temperature parameter
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

        epoch_losses = []
        epoch_early_losses = []
        epoch_full_losses = []
        epoch_diversity_losses = []
        epoch_agreement_losses = []
        epoch_confidences = []
        epoch_conf_stds = []

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
                epoch_losses.append(losses['total_loss'].item())
                epoch_early_losses.append(losses['early_loss'].item())
                epoch_full_losses.append(losses['full_loss'].item())
                epoch_diversity_losses.append(losses['diversity_loss'].item())
                epoch_agreement_losses.append(losses['agreement_loss'].item())
                epoch_confidences.append(losses['early_confidence'].item())
                epoch_conf_stds.append(losses['confidence_std'].item())

                # Update progress bar
                if batch_idx % 10 == 0:
                    pbar.set_postfix({
                        'loss': np.mean(epoch_losses[-50:]) if epoch_losses else 0,
                        'conf': np.mean(epoch_confidences[-50:]) if epoch_confidences else 0,
                        'std': np.mean(epoch_conf_stds[-50:]) if epoch_conf_stds else 0,
                        'temp': self.model.temperature.item()
                    })

            except RuntimeError as e:
                print(f"Error at batch {batch_idx}: {e}")
                continue

        # Store epoch metrics
        self.train_metrics['loss'].append(np.mean(epoch_losses))
        self.train_metrics['early_loss'].append(np.mean(epoch_early_losses))
        self.train_metrics['full_loss'].append(np.mean(epoch_full_losses))
        self.train_metrics['diversity_loss'].append(np.mean(epoch_diversity_losses))
        self.train_metrics['agreement_loss'].append(np.mean(epoch_agreement_losses))
        self.train_metrics['confidence'].append(np.mean(epoch_confidences))
        self.train_metrics['confidence_std'].append(np.mean(epoch_conf_stds))
        
    def validate(self, epoch):
        """Validate model"""
        self.model.eval()

        inference_times = []
        confidences = []
        exit_points = {'early': 0, 'full': 0}

        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc='Validation'):
                images = torch.stack([img.to(self.device) for img in images])

                # Measure inference time
                start_time = time.time()
                outputs = self.model(images)
                inference_time = (time.time() - start_time) * 1000

                inference_times.append(inference_time)
                if 'confidence' in outputs:
                    confidences.append(outputs['confidence'])

                # Track exit points
                if 'exit_point' in outputs:
                    exit_points[outputs['exit_point']] += 1

        # Calculate metrics
        total_samples = max(self.model.exit_stats['early'] + self.model.exit_stats['full'], 1)
        early_exit_rate = self.model.exit_stats['early'] / total_samples

        self.val_metrics['inference_time'].append(np.mean(inference_times))
        self.val_metrics['early_exit_rate'].append(early_exit_rate)
        self.val_metrics['avg_confidence'].append(np.mean(confidences) if confidences else 0)
        self.val_metrics['confidence_std'].append(np.std(confidences) if confidences else 0)

        # Store additional stats for analysis
        self.val_metrics['early_count'] = self.model.exit_stats['early']
        self.val_metrics['full_count'] = self.model.exit_stats['full']

        # Reset stats
        self.model.exit_stats = {'early': 0, 'full': 0}

        return early_exit_rate, np.mean(inference_times)
    
    def train(self, num_epochs):
        """Full training loop"""
        for epoch in range(num_epochs):
            print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
            
            # Train
            self.train_epoch(epoch)
            
            # Validate
            early_rate, inf_time = self.validate(epoch)
            
            # Update scheduler
            self.scheduler.step()
            
            # Print metrics
            print(f"Train Loss: {self.train_metrics['loss'][-1]:.4f}")
            print(f"  - Early Loss: {self.train_metrics['early_loss'][-1]:.4f}")
            print(f"  - Full Loss: {self.train_metrics['full_loss'][-1]:.4f}")
            print(f"  - Diversity Loss: {self.train_metrics['diversity_loss'][-1]:.4f}")
            print(f"Train Confidence: {self.train_metrics['confidence'][-1]:.3f} (±{self.train_metrics['confidence_std'][-1]:.3f})")
            print(f"Temperature: {self.model.temperature.item():.3f}")
            print(f"Early Exit Rate: {early_rate:.2%} (Early: {self.val_metrics['early_count']}, Full: {self.val_metrics['full_count']})")
            print(f"Avg Inference Time: {inf_time:.2f}ms")
            
            # Save checkpoint
            if (epoch + 1) % 25 == 0:
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
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def save_final_results(self):
        """Save final model and metrics"""
        # Save model
        final_model_path = os.path.join(MODEL_OUTPUT_PATH, 'early_exit_ssdlite_final.pth')
        torch.save(self.model.state_dict(), final_model_path)
        print(f"Final model saved: {final_model_path}")
        
        # Save metrics to CSV
        train_df = pd.DataFrame(self.train_metrics)
        train_df.to_csv(os.path.join(ANALYSIS_EE_OUTPUT_PATH, 'train_metrics.csv'), index=False)
        
        val_df = pd.DataFrame(self.val_metrics)
        val_df.to_csv(os.path.join(ANALYSIS_EE_OUTPUT_PATH, 'val_metrics.csv'), index=False)
        
        # Create and save plots
        self.create_plots()
        
    def create_plots(self):
        """Create training plots"""
        epochs = range(1, len(self.train_metrics['loss']) + 1)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Training losses
        axes[0, 0].plot(epochs, self.train_metrics['loss'], 'b-', label='Total Loss')
        axes[0, 0].plot(epochs, self.train_metrics['early_loss'], 'r--', label='Early Loss')
        axes[0, 0].plot(epochs, self.train_metrics['full_loss'], 'g--', label='Full Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Losses')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Confidence
        axes[0, 1].plot(epochs, self.train_metrics['confidence'], 'purple')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Average Confidence')
        axes[0, 1].set_title('Early Branch Confidence')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Early exit rate
        axes[0, 2].plot(epochs, self.val_metrics['early_exit_rate'], 'orange')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Early Exit Rate')
        axes[0, 2].set_title('Validation Early Exit Rate')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Inference time
        axes[1, 0].plot(epochs, self.val_metrics['inference_time'], 'green')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Time (ms)')
        axes[1, 0].set_title('Average Inference Time')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Validation confidence
        axes[1, 1].plot(epochs, self.val_metrics['avg_confidence'], 'cyan')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Confidence')
        axes[1, 1].set_title('Validation Average Confidence')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Combined metrics
        ax2 = axes[1, 2]
        ax2.plot(epochs, self.train_metrics['loss'], 'b-', label='Train Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        
        ax3 = ax2.twinx()
        ax3.plot(epochs, self.val_metrics['early_exit_rate'], 'r-', label='Early Exit Rate')
        ax3.set_ylabel('Early Exit Rate', color='r')
        ax3.tick_params(axis='y', labelcolor='r')
        
        axes[1, 2].set_title('Loss vs Early Exit Rate')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(ANALYSIS_EE_OUTPUT_PATH, 'training_plots.png'), dpi=150)
        plt.close()
        
        print(f"Plots saved to {ANALYSIS_EE_OUTPUT_PATH}")


def get_transform(train=False):
    """Get image transforms"""
    from torchvision import transforms

    if train:
        return transforms.Compose([
            transforms.Resize((320, 320)),  # Resize to fixed size for SSDLite320
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(0.5),
        ])
    return transforms.Compose([
        transforms.Resize((320, 320)),  # Resize to fixed size for SSDLite320
        transforms.ToTensor()
    ])


def main():
    """Main training function"""
    # Configuration
    BATCH_SIZE = 4  # Reduced for stability
    NUM_EPOCHS = 20  # Reduced for initial testing
    EXIT_THRESHOLD = 0.85  # Increased to force more full model usage
    NUM_WORKERS = 0  # Set to 0 for Windows to avoid multiprocessing issues

    print("=== Early Exit SSDLite Training for Chair Detection ===")
    print()

    # Clean and create working directories
    # clean_working_directory()
    create_output_directories()
    print()

    print(f"Configuration:")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Exit Threshold: {EXIT_THRESHOLD}")
    print(f"  Num Workers: {NUM_WORKERS}")
    print()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = ChairCocoDataset(
        TRAIN_IMAGES_PATH,
        os.path.join(ANNOTATIONS_PATH, 'instances_train2017.json'),
        transforms=get_transform(train=True),
        max_samples=5000  # Limit for faster training during testing
    )

    val_dataset = ChairCocoDataset(
        VAL_IMAGES_PATH,
        os.path.join(ANNOTATIONS_PATH, 'instances_val2017.json'),
        transforms=get_transform(train=False),
        max_samples=1000  # Limit for faster validation
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,  # Changed to 0 for Windows
        collate_fn=collate_fn,     # Using named function instead of lambda
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,  # Changed to 0 for Windows
        collate_fn=collate_fn,     # Using named function instead of lambda
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Initialize model
    print("\nInitializing model...")
    from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights
    base_model = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
    model = EarlyExitSSDLite(base_model, exit_threshold=EXIT_THRESHOLD)
    
    print(f"Model initialized with early exit after first feature block")
    print(f"Early exit threshold: {EXIT_THRESHOLD}")
    
    # Initialize trainer
    print("\nStarting training...")
    trainer = Trainer(model, train_loader, val_loader, device)
    
    # Train
    trainer.train(NUM_EPOCHS)
    
    print("\n=== Training Complete ===")
    print(f"Models saved to: {MODEL_OUTPUT_PATH}")
    print(f"Analysis saved to: {ANALYSIS_EE_OUTPUT_PATH}")
    
    # Final statistics
    final_metrics = {
        'final_train_loss': trainer.train_metrics['loss'][-1],
        'final_early_loss': trainer.train_metrics['early_loss'][-1],
        'final_full_loss': trainer.train_metrics['full_loss'][-1],
        'final_early_exit_rate': trainer.val_metrics['early_exit_rate'][-1],
        'final_inference_time': trainer.val_metrics['inference_time'][-1],
        'avg_confidence': trainer.train_metrics['confidence'][-1],
        'confidence_std': trainer.train_metrics['confidence_std'][-1],
        'final_temperature': model.temperature.item(),
        'early_count': trainer.val_metrics['early_count'],
        'full_count': trainer.val_metrics['full_count']
    }

    # Save final summary
    with open(os.path.join(ANALYSIS_EE_OUTPUT_PATH, 'final_summary.json'), 'w') as f:
        json.dump(final_metrics, f, indent=4)

    print("\nFinal Metrics:")
    print(f"  Train Loss: {final_metrics['final_train_loss']:.4f}")
    print(f"    - Early Loss: {final_metrics['final_early_loss']:.4f}")
    print(f"    - Full Loss: {final_metrics['final_full_loss']:.4f}")
    print(f"  Early Exit Rate: {final_metrics['final_early_exit_rate']:.2%}")
    print(f"    - Early Exits: {final_metrics['early_count']}")
    print(f"    - Full Model: {final_metrics['full_count']}")
    print(f"  Inference Time: {final_metrics['final_inference_time']:.2f}ms")
    print(f"  Avg Confidence: {final_metrics['avg_confidence']:.3f} (±{final_metrics['confidence_std']:.3f})")
    print(f"  Temperature: {final_metrics['final_temperature']:.3f}")


if __name__ == "__main__":
    main()