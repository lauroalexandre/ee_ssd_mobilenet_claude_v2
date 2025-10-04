# early_exit_ssdlite.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssd import SSD
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

# Create output directories
os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)
os.makedirs(ANALYSIS_EE_OUTPUT_PATH, exist_ok=True)

# COCO chair category ID
CHAIR_CATEGORY_ID = 62


class ChairCocoDataset(Dataset):
    """Dataset for chair detection only from COCO"""
    
    def __init__(self, img_folder, ann_file, transforms=None):
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
    
    def __init__(self, in_channels, num_classes=2):
        super().__init__()
        
        # Lightweight detection head
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        # Classification and regression heads
        # 4 anchors per location
        self.cls_head = nn.Conv2d(128, 4 * num_classes, kernel_size=1)
        self.reg_head = nn.Conv2d(128, 4 * 4, kernel_size=1)  # 4 bbox coords
        
    def forward(self, x):
        x = F.relu6(self.bn1(self.conv1(x)))
        x = F.relu6(self.bn2(self.conv2(x)))
        
        cls_logits = self.cls_head(x)
        bbox_regression = self.reg_head(x)
        
        return cls_logits, bbox_regression


class EarlyExitSSDLite(nn.Module):
    """SSDLite with early exit capability"""
    
    def __init__(self, base_model, exit_threshold=0.7):
        super().__init__()
        
        # Extract backbone features
        self.features_early = nn.Sequential(*list(base_model.backbone.body.children())[:12])
        self.features_late = nn.Sequential(*list(base_model.backbone.body.children())[12:])
        
        # Early exit branch after layer 12
        self.early_branch = EarlyExitBranch(in_channels=96, num_classes=2)
        
        # Original detection head (simplified for 2 classes)
        self.original_head = base_model.head
        self.anchor_generator = base_model.anchor_generator
        self.box_coder = base_model.box_coder
        self.postprocess = base_model.postprocess_detections
        
        self.exit_threshold = exit_threshold
        self.num_classes = 2
        
        # Statistics tracking
        self.exit_stats = {'early': 0, 'full': 0}
        
    def compute_confidence(self, cls_logits):
        """Compute confidence score for early exit decision"""
        batch_size = cls_logits.shape[0]
        cls_logits = cls_logits.permute(0, 2, 3, 1).contiguous()
        cls_logits = cls_logits.view(batch_size, -1, self.num_classes)
        
        # Apply softmax and get max confidence
        probs = F.softmax(cls_logits, dim=-1)
        max_probs = probs[:, :, 1].max(dim=1)[0]  # Max chair probability
        
        return max_probs
    
    def forward(self, images, targets=None):
        # Process early features
        features_early = self.features_early(images)
        
        # Early exit branch predictions
        early_cls, early_reg = self.early_branch(features_early)
        early_confidence = self.compute_confidence(early_cls)
        
        if self.training:
            # During training, compute both paths
            features_late = self.features_late(features_early)
            
            # Get full model predictions
            features = {'0': features_early, '1': features_late}
            detections = self.original_head(list(features.values()))
            
            # Compute losses
            if targets is not None:
                # Early branch loss
                early_loss = self.compute_detection_loss(
                    early_cls, early_reg, targets
                )
                
                # Full model loss
                full_loss = self.compute_detection_loss(
                    detections['cls_logits'], 
                    detections['bbox_regression'],
                    targets
                )
                
                # Combined loss with weighting
                alpha = 0.5
                total_loss = alpha * early_loss + (1 - alpha) * full_loss
                
                return {
                    'loss': total_loss,
                    'early_loss': early_loss,
                    'full_loss': full_loss,
                    'early_confidence': early_confidence.mean()
                }
            
            return detections
        
        else:
            # During inference, use early exit if confident
            avg_confidence = early_confidence.mean().item()
            
            if avg_confidence >= self.exit_threshold:
                self.exit_stats['early'] += 1
                # Format early predictions
                return self.format_predictions(early_cls, early_reg, images)
            else:
                self.exit_stats['full'] += 1
                # Continue with full model
                features_late = self.features_late(features_early)
                features = [features_early, features_late]
                detections = self.original_head(features)
                return self.format_predictions(
                    detections['cls_logits'],
                    detections['bbox_regression'],
                    images
                )
    
    def compute_detection_loss(self, cls_logits, bbox_regression, targets):
        """Simplified detection loss computation"""
        # This is a placeholder - implement proper SSD loss
        cls_loss = F.cross_entropy(
            cls_logits.view(-1, self.num_classes),
            torch.zeros(cls_logits.view(-1, self.num_classes).shape[0], dtype=torch.long).cuda()
        )
        
        reg_loss = F.smooth_l1_loss(
            bbox_regression,
            torch.zeros_like(bbox_regression)
        )
        
        return cls_loss + reg_loss
    
    def format_predictions(self, cls_logits, bbox_regression, images):
        """Format predictions for output"""
        # Simplified formatting - adapt based on actual requirements
        return {
            'cls_logits': cls_logits,
            'bbox_regression': bbox_regression
        }


class Trainer:
    """Training class for early exit model"""
    
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Optimizer with different learning rates
        self.optimizer = torch.optim.AdamW([
            {'params': model.early_branch.parameters(), 'lr': 1e-3},
            {'params': model.features_early.parameters(), 'lr': 1e-4},
            {'params': model.features_late.parameters(), 'lr': 1e-4},
            {'params': model.original_head.parameters(), 'lr': 5e-4}
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
        epoch_confidences = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for images, targets in pbar:
            images = torch.stack([img.to(self.device) for img in images])
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            losses = self.model(images, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Track metrics
            epoch_losses.append(losses['loss'].item())
            epoch_early_losses.append(losses['early_loss'].item())
            epoch_full_losses.append(losses['full_loss'].item())
            epoch_confidences.append(losses['early_confidence'].item())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': np.mean(epoch_losses[-50:]),
                'conf': np.mean(epoch_confidences[-50:])
            })
        
        # Store epoch metrics
        self.train_metrics['loss'].append(np.mean(epoch_losses))
        self.train_metrics['early_loss'].append(np.mean(epoch_early_losses))
        self.train_metrics['full_loss'].append(np.mean(epoch_full_losses))
        self.train_metrics['confidence'].append(np.mean(epoch_confidences))
        
    def validate(self, epoch):
        """Validate model"""
        self.model.eval()
        
        val_losses = []
        inference_times = []
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc='Validation'):
                images = torch.stack([img.to(self.device) for img in images])
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # Measure inference time
                start_time = time.time()
                predictions = self.model(images)
                inference_time = (time.time() - start_time) * 1000
                inference_times.append(inference_time)
        
        # Calculate metrics
        early_exit_rate = self.model.exit_stats['early'] / max(
            self.model.exit_stats['early'] + self.model.exit_stats['full'], 1
        )
        
        self.val_metrics['inference_time'].append(np.mean(inference_times))
        self.val_metrics['early_exit_rate'].append(early_exit_rate)
        
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
            print(f"Early Exit Rate: {early_rate:.2%}")
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
        
        # Loss plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
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
        axes[1, 0].plot(epochs, self.val_metrics['early_exit_rate'], 'orange')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Early Exit Rate')
        axes[1, 0].set_title('Validation Early Exit Rate')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Inference time
        axes[1, 1].plot(epochs, self.val_metrics['inference_time'], 'green')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Time (ms)')
        axes[1, 1].set_title('Average Inference Time')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(ANALYSIS_EE_OUTPUT_PATH, 'training_plots.png'), dpi=150)
        plt.close()
        
        print(f"Plots saved to {ANALYSIS_EE_OUTPUT_PATH}")


def get_transform(train=False):
    """Get image transforms"""
    from torchvision import transforms
    
    if train:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(0.5),
        ])
    return transforms.ToTensor()


def main():
    """Main training function"""
    # Configuration
    BATCH_SIZE = 8
    NUM_EPOCHS = 20
    EXIT_THRESHOLD = 0.7
    
    print("=== Early Exit SSDLite Training for Chair Detection ===\n")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = ChairCocoDataset(
        TRAIN_IMAGES_PATH,
        os.path.join(ANNOTATIONS_PATH, 'instances_train2017.json'),
        transforms=get_transform(train=True)
    )
    
    val_dataset = ChairCocoDataset(
        VAL_IMAGES_PATH,
        os.path.join(ANNOTATIONS_PATH, 'instances_val2017.json'),
        transforms=get_transform(train=False)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Initialize model
    print("\nInitializing model...")
    base_model = ssdlite320_mobilenet_v3_large(pretrained=True)
    model = EarlyExitSSDLite(base_model, exit_threshold=EXIT_THRESHOLD)
    
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
        'final_early_exit_rate': trainer.val_metrics['early_exit_rate'][-1],
        'final_inference_time': trainer.val_metrics['inference_time'][-1],
        'avg_confidence': trainer.train_metrics['confidence'][-1]
    }
    
    # Save final summary
    with open(os.path.join(ANALYSIS_EE_OUTPUT_PATH, 'final_summary.json'), 'w') as f:
        json.dump(final_metrics, f, indent=4)
    
    print("\nFinal Metrics:")
    print(f"  Train Loss: {final_metrics['final_train_loss']:.4f}")
    print(f"  Early Exit Rate: {final_metrics['final_early_exit_rate']:.2%}")
    print(f"  Inference Time: {final_metrics['final_inference_time']:.2f}ms")
    print(f"  Avg Confidence: {final_metrics['avg_confidence']:.3f}")


if __name__ == "__main__":
    main()