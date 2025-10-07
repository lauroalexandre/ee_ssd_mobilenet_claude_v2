# Comprehensive Analysis: Early Exit SSD MobileNet V3 Training Phases

**Date:** October 7, 2025
**Status:** Critical Analysis - Training Strategy Needs Complete Overhaul
**Phases Analyzed:** 1.1 through 1.9 (9 iterations)

---

## Executive Summary

After analyzing 9 training phases spanning multiple approaches to early-exit object detection, the results are **fundamentally flawed**. While some phases achieved fast inference times (5.8-8.2ms), the underlying model **failed to learn meaningful object detection**. The inference validation revealed that the model predicts almost uniformly across all spatial locations with confidence scores of 0.41-0.46, effectively behaving like an untrained network.

### Critical Finding
**The training approach uses random anchor-to-target assignment instead of IoU-based matching**, which is the fundamental cause of failure. This means the model never learned proper object localization or classification.

---

## Phase-by-Phase Analysis

### Overview Table

| Phase | Exit1 Rate | Exit2 Rate | Full Rate | Inference Time | Exit1 Conf | Exit2 Conf | Full Conf | Status |
|-------|------------|------------|-----------|----------------|------------|------------|-----------|---------|
| 1.1   | 0%         | 0%         | 100%      | 14.4ms        | 0.003      | 0.007      | 0.019     | ❌ Failed |
| 1.2   | 100%       | 0%         | 0%        | 8.1ms         | 0.478      | 0.529      | 0.634     | ⚠️ Fast but? |
| 1.3   | 0%         | 0%         | 100%      | 19.2ms        | 0.159      | 0.516      | 0.626     | ❌ Failed |
| 1.4   | 100%       | 0%         | 0%        | 8.2ms         | 0.476      | 0.482      | 0.593     | ⚠️ Fast but? |
| 1.5   | 0%         | 0%         | 100%      | 19.6ms        | 0.485      | 0.518      | 0.556     | ❌ Failed |
| 1.6   | 0%         | 0%         | 100%      | 21.1ms        | 0.329      | 0.722      | 0.792     | ❌ Failed |
| 1.7   | 100%       | 0%         | 0%        | 5.8ms         | 0.500      | 0.626      | 0.920     | ⚠️ Fast but? |
| 1.8   | 0%         | 0%         | 100%      | 14.5ms        | 0.000      | 0.000      | 0.529     | ❌ Failed |
| 1.9   | 0%         | 0%         | 100%      | 14.6ms        | 0.008      | 0.008      | 0.547     | ❌ Failed |

**Pattern:** Phases either exit 100% at Exit1 (fast) OR route 100% to full model (slow). Never achieved balanced distribution.

---

## What Went Wrong: The Fundamental Flaws

### 1. **CRITICAL: Random Anchor-Target Assignment**

**Location:** `prepare_targets_for_loss()` in all phase scripts

```python
# WRONG APPROACH - Used in all phases
num_positive = min(num_gt * 10, num_predictions // 20)  # 5% positive samples
positive_indices = torch.randperm(num_predictions, device=device)[:num_positive]

# Assign each positive anchor to a ground truth box
for j, idx in enumerate(positive_indices):
    gt_idx = j % num_gt  # Round-robin assignment ❌
    cls_targets[i, idx] = gt_labels[gt_idx]
    reg_targets[i, idx] = gt_boxes[gt_idx]
```

**Why This Is Fatal:**
- **No spatial correlation:** Random anchors assigned to objects regardless of location
- **No IoU matching:** Anchors that don't overlap with objects are marked as "positive"
- **Invalid supervision:** Bounding box regression gets nonsensical targets
- **Result:** Model learns to predict background uniformly everywhere

**What Should Have Been Done:**
```python
# CORRECT APPROACH - SSD-style anchor matching
ious = box_iou(anchors, gt_boxes)  # Compute all anchor-GT overlaps
best_gt_iou, best_gt_idx = ious.max(dim=1)

# Only mark anchors with high IoU as positive
positive_mask = best_gt_iou >= 0.5
matched_labels[positive_mask] = gt_labels[best_gt_idx[positive_mask]]
matched_boxes[positive_mask] = gt_boxes[best_gt_idx[positive_mask]]
```

### 2. **Incorrect Confidence Metric Design**

All phases used detection-quality-based confidence:

```python
# Phases 1.3-1.9 approach
objectness_threshold = 0.4  # or 0.5, 0.6, 0.7
min_confident_objects = 2   # or 3, 5

# Only consider "confident" if finds minimum objects above threshold
if num_confident_objects >= min_confident_objects:
    conf = 0.8 * obj_confidence + 0.2 * cls_confidence
else:
    conf = conf * 0.5  # Penalty ❌
```

**Why This Failed:**
- **Chicken-and-egg problem:** Early exits need good detections, but can't learn without exiting
- **Harsh penalties:** If model doesn't find N objects, confidence drops to near-zero
- **Training collapse:** Quality gates prevent model from ever practicing early exits
- **Evidence:** Phase 1.9 warmup showed confidence **crashed from 0.41 to 0.09** when gates activated

**Better Approach:**
```python
# Simple average probability - no quality gates during training
probs = F.softmax(cls_logits, dim=-1)
confidence = probs[:, 1].mean()  # Just average chair probability
```

### 3. **Oscillating Between Extremes**

The 9 phases show a clear pattern of overcorrection:

**Simple Phases (1.2, 1.4, 1.7):**
- Used basic threshold routing
- **Result:** 100% Exit1, but at least it's fast

**Complex Phases (1.5, 1.6, 1.8, 1.9):**
- Added quality controls, learned thresholds, distribution losses, warmup
- **Result:** 0% early exits, completely failed

**Pattern:** More complexity = worse results

### 4. **Temperature Instability**

| Phase | Temp1 | Temp2 | Temp_Full | Notes |
|-------|-------|-------|-----------|-------|
| 1.4   | 1.54  | 1.62  | 0.91      | Stable ✅ |
| 1.6   | 1.24  | 2.56  | 0.65      | Exit2 exploding |
| 1.7   | 24.56 | 1.71  | 0.59      | **Exit1 exploded!** |
| 1.9   | 1.17  | 3.00  | 0.94      | Exit2 maxed out |

**Why:** No clamping in early phases allowed unbounded growth during training.

### 5. **Never Validated Detection Quality**

**Critical Oversight:** All 9 phases measured:
- ✅ Inference time
- ✅ Exit distribution
- ✅ Confidence scores
- ❌ **Actual detection accuracy** (precision/recall)
- ❌ **mAP scores**
- ❌ **False positive rates**

**Result:** We optimized for speed without knowing if the model detects anything correctly.

---

## What Went Right (Partially)

### 1. **Architecture Design**
- Cascade structure with context flow between exits: **Good concept** ✅
- SE attention blocks in exit branches: **Helpful** ✅
- Multi-level early exits (layers 8, 12, full): **Well-positioned** ✅

### 2. **Exit Mechanism**
- Threshold-based routing in inference: **Works** ✅
- Confidence computation framework: **Solid foundation** ✅
- Statistics tracking: **Comprehensive** ✅

### 3. **Training Infrastructure**
- Data loading and augmentation: **Correct** ✅
- Loss weighting and scheduling: **Reasonable** ✅
- Knowledge distillation cascade: **Good idea** ✅
- Metrics tracking and visualization: **Excellent** ✅

### 4. **Phase Progression Logic**
- Identified that Phase 1.3's thresholds were too high: **Good analysis** ✅
- Added temperature clamping in later phases: **Right direction** ✅
- Tried warmup approach in Phase 1.9: **Creative** ✅

---

## Inference Validation Results (Phase 1.4)

### Actual Model Behavior

Running inference on 500 validation images revealed the truth:

**Classification Output:**
```
Chair probabilities:    0.41 - 0.46  (max: 0.4625)
Background probabilities: 0.54 - 0.59  (dominant everywhere)
```

**Detection Behavior:**
- Threshold 0.5: **0 detections** (original setting)
- Threshold 0.4: **2400 detections** (100% of anchors!)
- Threshold 0.35: **~460 detections per image after NMS**

**Bounding Box Regression:**
```
Delta values: dx=182-377, dy=184-379, dw=176-379, dh=186-378
Expected:     dx=-2 to +2 (small offsets from anchors)
```
→ Model never learned valid bbox regression

**Conclusion:** The model predicts almost uniformly across the entire image, essentially behaving like an **untrained network**.

---

## Root Cause Analysis

### The Training Was Doomed From The Start

All 9 phases share the same fatal flaw:

```
Random Anchor Assignment
         ↓
Model Gets Nonsensical Supervision
         ↓
Learns to Predict Background Everywhere
         ↓
Confidence Scores: 0.41-0.46 (uniform)
         ↓
No Meaningful Object Detection
```

**Evidence:**
1. **Uniform confidence:** All predictions 0.41-0.46 → no selectivity learned
2. **Invalid bbox regression:** Values 175-379 instead of -2 to +2
3. **No localization:** ~460 detections per image at fixed grid locations
4. **Background dominant:** 0.54-0.59 probability everywhere

### Why Quality Gates Made It Worse

Phases with quality-based confidence (1.3, 1.5, 1.6, 1.8, 1.9):

```
Model predicts ~0.45 everywhere
         ↓
Quality gate requires 2-3 objects above 0.5-0.7
         ↓
Model finds 0 confident objects
         ↓
Confidence multiplied by penalty: 0.45 * 0.5 = 0.225
         ↓
Never exceeds exit threshold
         ↓
0% early exits → no practice → never improves
```

**Phase 1.9 Warmup Evidence:**
- Epochs 1-3 (no gates): Exit1 conf = 0.41, **100% Exit1**, 6ms ✅
- Epoch 4 (gates ON): Confidence crashes to **0.09** ❌
- Epochs 5-20: Confidence stays at **0.006-0.008** ❌

This proves the quality gates **actively prevented** the model from working.

---

## Lessons Learned

### 1. **Fundamental First, Optimization Second**
- ❌ We tried to optimize early-exit routing before the model could detect objects
- ✅ Should have: Validate detection quality first, then add early exits

### 2. **Complexity Is Not Progress**
- Simple phases (1.2, 1.4) at least ran fast
- Complex phases (1.5-1.9) added sophistication but made things worse
- **Pattern:** Each "improvement" moved further from success

### 3. **Validation Is Critical**
- We tracked metrics but never checked if the model actually detects chairs
- All 9 phases could have been avoided with one early detection quality check

### 4. **Don't Trust Confidence Scores Alone**
- Confidence of 0.476 looked reasonable
- Actual behavior: uniform prediction everywhere
- **Lesson:** Always inspect raw outputs and detection visualizations

### 5. **Architecture vs. Training**
- The early-exit architecture is sound
- The training methodology is fundamentally broken
- **Fixing training is the blocker, not architecture**

---

## Why This Happened: The Human Factor

### 1. **Assumption Error**
We assumed the model was learning to detect objects because:
- Training losses decreased
- Confidence scores looked reasonable (0.4-0.6)
- The code ran without errors

**Reality:** The model was just learning to predict background with slightly more confidence.

### 2. **Incremental Trap**
Each phase built on the previous one's code:
- Phase 1.1 → 1.2 → ... → 1.9
- The random anchor assignment bug was in ALL phases
- We kept "improving" a fundamentally broken foundation

### 3. **Missing Ground Truth**
Never validated against:
- Actual chair locations in images
- Precision/recall metrics
- Visual inspection of detections

**Result:** Optimizing for the wrong metrics (speed and exit distribution) while detection quality was zero.

### 4. **Complexity Bias**
When simple approaches (1.2, 1.4) succeeded, we thought:
- "They're too simple, let's add quality controls"
- "Need balanced exits, not 100% Exit1"

**Reality:** The simple approaches "worked" only because they bypassed quality checks. Adding quality controls exposed the broken detection.

---

## The Path Forward: Complete Redesign Required

### Phase 1: Fix The Foundation (Weeks 1-2)

#### 1.1 Implement Proper SSD Anchor Matching

**Create new file:** `ssd_anchor_matching.py`

```python
def match_anchors_to_targets(anchors, targets, iou_threshold=0.5):
    """
    Proper SSD-style anchor matching

    For each ground truth box:
    1. Find anchors with IoU >= threshold
    2. Assign GT box to those anchors
    3. Mark other anchors as background
    """
    gt_boxes = targets['boxes']
    gt_labels = targets['labels']

    # Compute IoU matrix: [num_anchors, num_gt]
    ious = box_iou(anchors, gt_boxes)

    # For each anchor, find best matching GT
    best_gt_iou, best_gt_idx = ious.max(dim=1)

    # Initialize all as background (label 0)
    matched_labels = torch.zeros(len(anchors), dtype=torch.long)
    matched_boxes = torch.zeros((len(anchors), 4), dtype=torch.float32)

    # Mark high-IoU anchors as positive
    positive_mask = best_gt_iou >= iou_threshold
    matched_labels[positive_mask] = gt_labels[best_gt_idx[positive_mask]]
    matched_boxes[positive_mask] = gt_boxes[best_gt_idx[positive_mask]]

    return matched_labels, matched_boxes, best_gt_iou
```

**Action:** Replace `prepare_targets_for_loss()` in training script with this proper matching.

#### 1.2 Train Baseline SSD First

**Goal:** Get a working single-class chair detector before adding early exits.

```python
# Remove all early exit logic
# Train just the full model with proper anchor matching
# Validate detection quality: target mAP > 0.40
```

**Success Criteria:**
- mAP@0.5 > 0.40 on validation set
- Precision > 0.60, Recall > 0.50
- Visual inspection: correctly detects chairs

#### 1.3 Validate Detection Quality

**Create validation script:**
```python
def compute_detection_metrics(predictions, ground_truth):
    """
    Compute:
    - Precision, Recall, F1
    - mAP@0.5, mAP@0.75
    - False positive rate
    - Localization error (IoU)
    """
    # Use standard COCO evaluation metrics
    pass
```

**Do NOT proceed to early exits until baseline achieves acceptable detection quality.**

---

### Phase 2: Add Early Exits Correctly (Weeks 3-4)

#### 2.1 Start With Simplest Possible Approach

**No quality gates. No learned thresholds. Just basic routing.**

```python
def compute_confidence(cls_logits, temperature):
    """
    Ultra-simple confidence: average foreground probability
    NO detection counting, NO quality penalties
    """
    probs = F.softmax(cls_logits / temperature, dim=-1)
    chair_probs = probs[:, :, 1]  # Chair class
    return chair_probs.mean()

# Exit logic
if compute_confidence(exit1_cls, temp1) >= 0.60:  # Fixed threshold
    return exit1_predictions
```

**Rationale:** If this doesn't work, the architecture is wrong. If it does work, optimize from there.

#### 2.2 Measure Detection Quality At Each Exit

**Critical validation:**
```python
# For each image:
exit1_predictions = model.exit1(image)
exit2_predictions = model.exit2(image)
full_predictions = model.full(image)

# Compare against ground truth:
exit1_map = compute_map(exit1_predictions, ground_truth)
exit2_map = compute_map(exit2_predictions, ground_truth)
full_map = compute_map(full_predictions, ground_truth)

print(f"mAP: Exit1={exit1_map:.3f}, Exit2={exit2_map:.3f}, Full={full_map:.3f}")
```

**Decision criteria:**
- If Exit1 mAP < 0.30: Too early, move to layer 10 or 12
- If Exit1 mAP > 0.35: Acceptable for speed/accuracy tradeoff
- If Exit1 mAP > Full mAP: Something is very wrong

#### 2.3 Accept Reality

**Possible outcomes:**

**Outcome A: Early exits work well**
- Exit1 mAP = 0.35, Full mAP = 0.45
- Speed gain: 2-3x
- **Action:** Ship it. This is a good tradeoff.

**Outcome B: Early exits are too weak**
- Exit1 mAP = 0.15, Full mAP = 0.45
- Speed gain exists but quality too low
- **Action:** Move exits later (layers 12, 14) or accept limitation

**Outcome C: No speed gain**
- All images need full model for acceptable quality
- **Action:** Early exits aren't viable for this task

---

### Phase 3: Careful Optimization (Week 5+)

**Only if Phase 2 succeeds and baseline detection quality is good:**

#### 3.1 Add Temperature Clamping
```python
self.temperature1 = nn.Parameter(torch.ones(1) * 1.5, requires_grad=True)
# In training step:
self.temperature1.data.clamp_(0.5, 3.0)  # Prevent explosion
```

#### 3.2 Try Learned Thresholds (Optional)
```python
self.exit1_threshold = nn.Parameter(torch.tensor(0.60), requires_grad=True)
# Only if fixed thresholds don't give balanced distribution
```

#### 3.3 Consider Adaptive Routing (Advanced)
```python
# Route based on predicted detection difficulty
# But ONLY after basic approach works
```

**Do NOT add complexity unless:**
1. Basic approach works
2. Have specific problem to solve
3. Can measure improvement objectively

---

## Alternative Architectures To Consider

### Option A: Later Exit Points

Current exits: Layer 8 (early), Layer 12 (mid), Full (late)

**Problem:** Layer 8 might be too early for reliable detection

**Alternative:**
- Exit 1: Layer 12 (instead of 8)
- Exit 2: Layer 14 (instead of 12)
- Full: Complete model

**Trade-off:** Less speed gain, but might actually work

### Option B: Confidence-Based Duplication

Instead of different exit points, same detection quality with early stopping:

```python
# Run Exit1 predictions
if max_confidence(exit1) >= 0.8:  # Very confident
    return exit1_predictions
else:
    # Continue to full model
    # Exit1 was a "failed attempt" - happens rarely
```

**Advantage:** No quality degradation when using early exit

### Option C: Hybrid Detection + Classification

```python
# Always run full detection (find boxes)
# Early exit only for classification refinement
# All boxes come from full model, some classes from early model
```

**Advantage:** Localization quality guaranteed

### Option D: Accept Single Exit

Based on Phase 1.4's behavior:

```python
# Fact: 100% of images exit at Exit1
# If detection quality at Exit1 is acceptable (mAP > 0.35)
# Then this IS the solution - one simple early exit

# Don't force balanced distribution if it's not natural
```

**Advantage:** Simple, fast, might be optimal

---

## Recommended Next Steps (Priority Order)

### 1. **STOP all current training immediately**

Do not run Phase 1.10, 1.11, etc. The foundation is broken.

### 2. **Validate Phase 1.4 detection quality**

```bash
# Load Phase 1.4 model
# Run on validation set with ground truth
# Compute: precision, recall, mAP, false positive rate
# Visual inspection: do detected boxes match actual chairs?
```

**Decision point:**
- If mAP > 0.35: Phase 1.4 might be acceptable as-is
- If mAP < 0.20: Need complete retraining with proper anchor matching

### 3. **Create baseline SSD detector**

```python
# New script: baseline_ssd_chair_detector.py
# Train full model only (no early exits)
# Use PROPER anchor matching
# Target: mAP > 0.40 before proceeding
```

**Timeline:** 1-2 days to implement, 1-2 days to train

### 4. **Add early exits to working baseline**

**Only after Step 3 succeeds:**
```python
# Start with simplest possible routing
# No quality gates during training
# Measure detection quality at each exit
```

**Timeline:** 2-3 days to implement, 2-3 days to train

### 5. **Document and share findings**

```markdown
# Share with advisor/team:
- "We discovered the training was fundamentally flawed"
- "Here's what we learned"
- "Here's the corrected approach"
- "Expected timeline: 1-2 weeks to proper baseline"
```

---

## Technical Debt Summary

### Must Fix
1. ❌ Random anchor assignment → IoU-based matching
2. ❌ No detection quality validation → Add mAP computation
3. ❌ Quality-based confidence during training → Simple average probability
4. ❌ Temperature unbounded → Clamp to [0.5, 3.0]

### Should Fix
5. ⚠️ No hard negative mining → Add in baseline training
6. ⚠️ No focal loss → Consider for class imbalance
7. ⚠️ Fixed threshold routing → Could learn thresholds later

### Nice To Have
8. 💡 Visualize attention maps at each exit
9. 💡 Analyze which images benefit from early exit
10. 💡 Compare with other early-exit papers

---

## Estimated Timeline for Correct Implementation

### Week 1: Foundation
- Days 1-2: Implement proper anchor matching
- Days 3-4: Train baseline SSD (no early exits)
- Day 5: Validate detection quality (mAP, precision, recall)

**Gate:** Do NOT proceed if mAP < 0.40

### Week 2: Early Exits
- Days 1-2: Add early exit branches
- Days 3-4: Train with simple routing
- Day 5: Measure quality at each exit

**Gate:** If Exit1 mAP < 0.30, reposit exit point

### Week 3: Optimization (if needed)
- Days 1-2: Add temperature clamping
- Days 3-4: Tune thresholds for balanced distribution
- Day 5: Final validation and comparison

**Total:** 2-3 weeks to proper working system

---

## Conclusion: The Harsh Truth

### What We Learned (The Hard Way)

After 9 training phases and significant compute time:

1. **The model never learned to detect objects** - random anchor assignment broke everything
2. **Confidence scores were misleading** - model predicted uniformly everywhere
3. **Complexity made things worse** - simple broken approach better than complex broken approach
4. **We optimized the wrong metrics** - speed without quality is meaningless
5. **Validation is not optional** - should have checked detection quality immediately

### The Silver Lining

**We now know:**
- ✅ The early-exit architecture design is sound
- ✅ The inference system works correctly
- ✅ The training infrastructure is solid
- ✅ We have excellent metrics tracking
- ✅ We understand what went wrong and how to fix it

**The problem is fixable** - it requires proper anchor matching and detection quality validation, not architectural changes.

### Final Recommendation

**Option 1: Quick Win (1 week)**
- Validate Phase 1.4 detection quality
- If mAP > 0.35, accept it as "fast but lower quality" mode
- Document limitations and ship

**Option 2: Proper Solution (2-3 weeks)**
- Implement correct SSD training from scratch
- Add early exits to working baseline
- Achieve 2x speedup with <10% mAP drop

**Option 3: Research Pivot**
- Acknowledge early exits may not be viable for multi-object detection
- Focus on other optimizations (pruning, quantization, NAS)
- Cite this as "negative result" in dissertation

**Do NOT:** Continue sequential training phases (1.10, 1.11, ...) without fixing the foundation.

---

**Author:** AI Training Analysis System
**Date:** October 7, 2025
**Status:** Report Complete - Awaiting Decision on Path Forward

---

## Appendix: Code Snippets for Fixes

### A. Proper Anchor Matching Implementation

```python
def box_iou_torch(boxes1, boxes2):
    """Compute IoU between all pairs of boxes"""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-6)
    return iou

def generate_anchors_grid(feature_height, feature_width, stride, anchor_sizes):
    """Generate anchor boxes at grid positions"""
    anchors = []
    for h in range(feature_height):
        for w in range(feature_width):
            cx = (w + 0.5) * stride
            cy = (h + 0.5) * stride
            for size in anchor_sizes:
                anchors.append([
                    cx - size/2,  # x1
                    cy - size/2,  # y1
                    cx + size/2,  # x2
                    cy + size/2   # y2
                ])
    return torch.tensor(anchors)

def match_anchors_proper(anchors, gt_boxes, gt_labels, iou_threshold=0.5):
    """Proper SSD anchor matching"""
    if len(gt_boxes) == 0:
        # No ground truth - all background
        return torch.zeros(len(anchors), dtype=torch.long), \
               torch.zeros((len(anchors), 4)), \
               torch.zeros(len(anchors))

    # Compute IoU
    ious = box_iou_torch(anchors, gt_boxes)
    best_iou, best_gt_idx = ious.max(dim=1)

    # Assign labels
    matched_labels = torch.zeros(len(anchors), dtype=torch.long)
    matched_boxes = torch.zeros((len(anchors), 4))
    matched_ious = torch.zeros(len(anchors))

    positive_mask = best_iou >= iou_threshold
    matched_labels[positive_mask] = gt_labels[best_gt_idx[positive_mask]]
    matched_boxes[positive_mask] = gt_boxes[best_gt_idx[positive_mask]]
    matched_ious[positive_mask] = best_iou[positive_mask]

    return matched_labels, matched_boxes, matched_ious
```

### B. Simple Confidence Computation

```python
def compute_confidence_simple(cls_logits, temperature):
    """
    Simplest possible confidence metric
    No quality gates, no detection counting
    Just average foreground probability
    """
    # cls_logits: [B, num_anchors*num_classes, H, W]
    B, _, H, W = cls_logits.shape
    num_anchors = 6
    num_classes = 2

    # Reshape to [B, H*W*num_anchors, num_classes]
    logits = cls_logits.permute(0, 2, 3, 1).contiguous()
    logits = logits.view(B, H * W * num_anchors, num_classes)

    # Apply temperature scaling and softmax
    probs = F.softmax(logits / temperature, dim=-1)

    # Get chair class probability (index 1)
    chair_probs = probs[:, :, 1]

    # Average across all anchors (per image in batch)
    confidence = chair_probs.mean(dim=1)

    return confidence
```

### C. Detection Quality Validation

```python
def evaluate_detection_quality(model, dataloader, device, iou_threshold=0.5, conf_threshold=0.5):
    """
    Compute detection quality metrics
    Returns: precision, recall, mAP, false_positive_rate
    """
    model.eval()

    all_predictions = []
    all_ground_truths = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)

            # Get predictions
            outputs = model(images)
            predictions = decode_predictions(outputs, conf_threshold)

            all_predictions.extend(predictions)
            all_ground_truths.extend(targets)

    # Compute metrics
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for pred, gt in zip(all_predictions, all_ground_truths):
        pred_boxes = pred['boxes']
        gt_boxes = gt['boxes']

        if len(pred_boxes) == 0 and len(gt_boxes) == 0:
            continue
        elif len(pred_boxes) == 0:
            false_negatives += len(gt_boxes)
        elif len(gt_boxes) == 0:
            false_positives += len(pred_boxes)
        else:
            # Compute IoU matrix
            ious = box_iou_torch(pred_boxes, gt_boxes)

            # Match predictions to ground truth
            for i in range(len(pred_boxes)):
                if ious[i].max() >= iou_threshold:
                    true_positives += 1
                else:
                    false_positives += 1

            # Count missed ground truths
            matched_gt = (ious.max(dim=0)[0] >= iou_threshold).sum()
            false_negatives += len(gt_boxes) - matched_gt

    precision = true_positives / (true_positives + false_positives + 1e-6)
    recall = true_positives / (true_positives + false_negatives + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }
```

---

## References for Proper Implementation

1. **Original SSD Paper:** Liu et al., "SSD: Single Shot MultiBox Detector" (ECCV 2016)
   - Section 2.2: Matching strategy
   - Section 2.3: Training objective

2. **Focal Loss:** Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)
   - Better than standard cross-entropy for object detection

3. **Early Exit Networks:** Teerapittayanon et al., "BranchyNet" (ICPR 2016)
   - Proper early exit training methodology

4. **Detection Metrics:** "COCO Detection Challenge"
   - Standard evaluation protocols

---

**END OF REPORT**
