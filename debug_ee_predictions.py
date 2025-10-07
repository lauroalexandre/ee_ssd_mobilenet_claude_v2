#!/usr/bin/env python3
"""Debug script to inspect early-exit model predictions"""

import torch
import torch.nn.functional as F
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
import sys

# Import the model architecture from the inference script
sys.path.insert(0, '.')
from baseline_inference_complete_ee_phase_1_4 import MultiLevelCascadeEarlyExitSSDLite

def debug_model_predictions():
    """Debug what the model actually predicts"""

    MODEL_PATH = 'working/model_trained/early_exit_ssdlite_phase1_4_final.pth'
    IMAGE_PATH = 'subset_validation/images/000000000139.jpg'

    print("="*80)
    print("DEBUG: Early Exit Model Predictions")
    print("="*80)

    # Load model
    print("\n1. Loading model...")
    base_model = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
    model = MultiLevelCascadeEarlyExitSSDLite(base_model, exit1_threshold=0.45, exit2_threshold=0.60)

    state_dict = torch.load(MODEL_PATH, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("   Model loaded successfully")

    # Load and preprocess image
    print(f"\n2. Loading image: {IMAGE_PATH}")
    image = Image.open(IMAGE_PATH).convert('RGB')
    print(f"   Original size: {image.size}")

    image_resized = image.resize((320, 320), Image.BILINEAR)
    image_tensor = TF.to_tensor(image_resized).unsqueeze(0)
    print(f"   Tensor shape: {image_tensor.shape}")

    # Run inference
    print("\n3. Running inference...")
    with torch.no_grad():
        outputs = model(image_tensor)

    print(f"   Exit point: {outputs['exit_point']}")
    print(f"   Confidence: {outputs['confidence']:.4f}")

    # Inspect raw outputs
    cls_logits = outputs['cls_logits']
    bbox_regression = outputs['bbox_regression']

    print(f"\n4. Raw model outputs:")
    print(f"   cls_logits shape: {cls_logits.shape}")
    print(f"   bbox_regression shape: {bbox_regression.shape}")

    # Analyze classification outputs
    print(f"\n5. Classification analysis:")
    h, w = cls_logits.shape[2], cls_logits.shape[3]
    num_anchors = 6
    num_classes = 2

    print(f"   Feature map size: {h}x{w}")
    print(f"   Total predictions: {h * w * num_anchors}")

    # Reshape logits
    cls_logits_reshaped = cls_logits.permute(0, 2, 3, 1).contiguous()
    cls_logits_reshaped = cls_logits_reshaped.view(1, h * w * num_anchors, num_classes)

    # Get probabilities
    probs = F.softmax(cls_logits_reshaped[0], dim=-1)
    chair_probs = probs[:, 1]  # Chair class (index 1)
    bg_probs = probs[:, 0]      # Background class (index 0)

    print(f"\n6. Probability statistics:")
    print(f"   Chair probabilities:")
    print(f"     Min:  {chair_probs.min().item():.4f}")
    print(f"     Max:  {chair_probs.max().item():.4f}")
    print(f"     Mean: {chair_probs.mean().item():.4f}")
    print(f"     Std:  {chair_probs.std().item():.4f}")

    print(f"\n   Background probabilities:")
    print(f"     Min:  {bg_probs.min().item():.4f}")
    print(f"     Max:  {bg_probs.max().item():.4f}")
    print(f"     Mean: {bg_probs.mean().item():.4f}")
    print(f"     Std:  {bg_probs.std().item():.4f}")

    # Check different thresholds
    print(f"\n7. Detection counts at different thresholds:")
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        count = (chair_probs >= threshold).sum().item()
        print(f"   Threshold {threshold:.1f}: {count} detections ({count/(h*w*num_anchors)*100:.1f}%)")

    # Top predictions
    print(f"\n8. Top 10 chair predictions:")
    top_scores, top_indices = torch.topk(chair_probs, min(10, len(chair_probs)))
    for i, (score, idx) in enumerate(zip(top_scores, top_indices)):
        print(f"   #{i+1}: score={score.item():.4f}, index={idx.item()}")

    # Bbox regression statistics
    print(f"\n9. Bbox regression statistics:")
    bbox_flat = bbox_regression.view(-1, 4)
    for coord_idx, coord_name in enumerate(['dx', 'dy', 'dw', 'dh']):
        values = bbox_flat[:, coord_idx]
        print(f"   {coord_name}: min={values.min().item():.4f}, max={values.max().item():.4f}, mean={values.mean().item():.4f}")

    print("\n" + "="*80)
    print("DEBUG COMPLETE")
    print("="*80)

if __name__ == '__main__':
    debug_model_predictions()
