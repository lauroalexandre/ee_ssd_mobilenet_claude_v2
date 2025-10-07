# Early Exit Inference Analysis - Phase 1.4

## Issue Summary

The trained model from `early_exit_training_system_v3_phase1_4.py` did not converge to produce meaningful object detections. This document explains the findings and the workaround implemented.

## Root Cause Analysis

### 1. Model Training Issues

The Phase 1.4 training script used a **simplified target assignment approach** that did not properly match ground truth objects to anchor predictions:

```python
# From training script line 636-648
# Randomly select anchors to assign as positive
positive_indices = torch.randperm(num_predictions, device=device)[:num_positive]

# Assign each positive anchor to a ground truth box
for j, idx in enumerate(positive_indices):
    gt_idx = j % num_gt  # Round-robin assignment
    cls_targets[i, idx] = gt_labels[gt_idx]
```

This **random assignment** instead of IoU-based matching meant the model never learned proper anchor-to-object correspondence.

### 2. Model Output Characteristics

Debug analysis revealed:

**Classification Outputs:**
- Chair class probabilities: **0.41 - 0.46** (max: 0.4625)
- Background probabilities: **0.54 - 0.59** (dominant)
- **Result:** Model predicts background everywhere, never reaching original threshold of 0.5

**Bounding Box Regression:**
- Delta values: **175 - 379** (should be small values like -2 to +2)
- **Result:** Invalid bbox predictions that produce boxes outside the image

### 3. Detection Statistics

At different confidence thresholds:
- **Threshold 0.5:** 0 detections (original setting)
- **Threshold 0.4:** 2400 detections (100% of anchors)
- **Threshold 0.35:** ~460-470 detections per image after NMS

## Implemented Workaround

To generate **any** detections from this model, the following modifications were made:

### 1. Lowered Confidence Threshold
```python
CONFIDENCE_THRESHOLD = 0.35  # Changed from 0.5
```

### 2. Fixed-Size Box Decoding
Since bbox regression didn't learn meaningful values, boxes are generated as:
- **Position:** Anchor center location (ignoring large bbox deltas)
- **Size:** Fixed anchor sizes (40, 60, 80, 100, 120, 140 pixels)

```python
# Ignore learned bbox regression (values are invalid)
pred_cx = filtered_centers_x + filtered_bbox_deltas[:, 0].clamp(-2, 2)
pred_cy = filtered_centers_y + filtered_bbox_deltas[:, 1].clamp(-2, 2)
pred_w = filtered_sizes.clone()  # Use anchor size directly
pred_h = filtered_sizes.clone()
```

## Current Inference Behavior

### What the Model Does:
- Detects ~460-470 "chairs" per image (at almost every anchor location)
- All detections have confidence scores between **0.35 - 0.46**
- Boxes are fixed-size rectangles at grid positions
- No learned selectivity - detects everywhere

### Detection Quality:
- ⚠️ **Low quality:** Not selective, predicts almost everywhere
- ⚠️ **Fixed sizes:** Cannot adapt to actual object sizes
- ⚠️ **No localization:** Boxes don't align with actual objects
- ✅ **Consistent:** All images processed with bounding boxes visible

## Output Files Generated

```
subset_validation/ee_result_phase_1_4/
├── ee_results_phase_1_4.json          # Full results with ~460 dets/image
├── images_phase_1_4/                  # Annotated images with boxes
├── inference_analysis_phase_1_4.png   # Performance plots
└── cpu_vs_gpu_comparison_phase_1_4.png (if GPU available)
```

## Recommendations

### For Proper Detection Model:

1. **Fix Training Target Assignment:**
   - Use IoU-based anchor matching (like standard SSD)
   - Match each ground truth to best overlapping anchors
   - Only mark high-IoU anchors as positive

2. **Proper Loss Functions:**
   - Use focal loss or hard negative mining
   - Weight classification and regression losses appropriately
   - Monitor training convergence carefully

3. **Validation During Training:**
   - Check actual detection outputs during training
   - Verify confidence score distributions
   - Validate bbox regression learning

### For Current Model:

The model can be used for:
- ✅ **Speed benchmarking** (inference time, early exit behavior)
- ✅ **Architecture testing** (model structure, exit points)
- ✅ **Visualization** (showing what untrained/poorly-trained model outputs)
- ❌ **NOT for actual object detection** (quality is too low)

## Technical Details

### Model Architecture:
- Base: SSDLite320 MobileNetV3 Large
- Exit points: Layer 8 (exit1), Layer 12 (exit2), Full model
- Feature map size: 20x20 (exit1)
- Anchors per location: 6
- Total predictions: 2400

### Exit Behavior:
- **100% exit at exit1** (confidence 0.475 > threshold 0.45)
- Never reaches exit2 or full model
- Early exit system works architecturally, but model didn't train

## Conclusion

The inference code now successfully:
1. ✅ Loads the trained model
2. ✅ Processes all images
3. ✅ Generates bounding boxes on images
4. ✅ Creates analysis reports and visualizations

However, the **detection quality is poor** due to fundamental training issues. The model behaves like an untrained network, predicting background everywhere with low confidence.

For production use, the training approach needs to be fixed to use proper SSD-style anchor matching and loss functions.

---

**Date:** October 2025
**Model:** early_exit_ssdlite_phase1_4_final.pth
**Status:** Inference working, training needs improvement
