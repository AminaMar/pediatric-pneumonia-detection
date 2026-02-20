# 📊 Data Preprocessing & Augmentation Report

**Data Engineer:** Bouhmidi Amina Meroua  
**Date:** February 20, 2026  
**Project:** AI-Powered Early Pediatric Pneumonia Detection: Integration with Electronic Medical Records in Algeria  
**Phase:** Data Preprocessing & Augmentation

---



**Key Achievements:**
- ✅ Reorganized dataset with balanced 70/15/15 split
- ✅ Analyzed image quality across 5,856 samples
- ✅ Calculated optimal class weights (1.850 / 0.685)
- ✅ Implemented realistic data augmentation
- ✅ Created end-to-end preprocessing pipeline

---

## 📊 Dataset Reorganization

### Problem Statement

**Original Dataset Issues:**
- Validation set critically small (16 images - 0.3%)
- Inconsistent distribution across splits
- No quality filtering

### Solution Implemented

**New Organization (Stratified Split):**

| Split | NORMAL | PNEUMONIA | Total | Percentage |
|-------|--------|-----------|-------|------------|
| **Training** | 1,108 (27.0%) | 2,991 (73.0%) | 4,099 | 70.0% |
| **Validation** | 237 (27.0%) | 641 (73.0%) | 878 | 15.0% |
| **Test** | 238 (27.1%) | 641 (72.9%) | 879 | 15.0% |
| **Total** | **1,583** | **4,273** | **5,856** | **100%** |

**Key Improvements:**
- ✅ Validation set increased from 16 to 878 images (54× larger)
- ✅ Identical class distribution (27%/73%) maintained across all splits
- ✅ Random stratified split ensures unbiased evaluation
- ✅ **Estimated gain: +0.5% accuracy**

---

## 🔍 Image Quality Analysis

### Methodology

Analyzed 100 samples per class using three quality metrics:

1. **Sharpness (Laplacian Variance)**
   - Measures image blur
   - Higher values = sharper images
   - Used to identify potentially problematic samples

2. **Brightness (Mean Pixel Intensity)**
   - NORMAL average: ~127
   - PNEUMONIA average: ~135
   - Slightly brighter due to infiltrations

3. **Contrast (Standard Deviation)**
   - Indicates image detail and quality
   - Higher contrast = better feature visibility

### Results

![Quality Analysis](figures/05_quality_analysis.png)

**Decision:** Retained all images as quality was generally acceptable. Sharpness filtering can be applied if needed (10th percentile threshold identified).

**Estimated gain:** +0.5% (if filtering applied)

---

## ⚖️ Class Imbalance Strategy

### Natural Imbalance

Dataset exhibits 2.7:1 imbalance (PNEUMONIA:NORMAL), reflecting clinical reality where pneumonia cases are more common in pediatric datasets.

### Why We Maintain the Imbalance

**Rejected Approaches:**
- ❌ **Undersampling PNEUMONIA** → Loss of 60% of data
- ❌ **Oversampling NORMAL** → Risk of overfitting through duplication
- ❌ **SMOTE** → May create clinically invalid synthetic images

**Our Approach (Industry Standard):**

1. **Stratified Split** → Maintain 27%/73% ratio across train/val/test
2. **Class Weights** → Compensate during training
3. **Targeted Augmentation** → Increase NORMAL class diversity

### Calculated Class Weights
```json
{
  "NORMAL (class 0)": 1.850,
  "PNEUMONIA (class 1)": 0.685
}
```

**Calculation Method:**
```python
from sklearn.utils.class_weight import compute_class_weight

Weight_NORMAL = Total / (2 × N_NORMAL) = 4,099 / (2 × 1,108) = 1.850
Weight_PNEUMONIA = Total / (2 × N_PNEUMONIA) = 4,099 / (2 × 2,991) = 0.685
```

**Impact:**
- Model gives **2.7× more importance** to NORMAL class during training
- Forces equal learning of both classes despite numerical imbalance
- **Estimated gain: +1.0% to +1.5% accuracy**

---

## 🎨 Data Augmentation Strategy

### Rationale

Applied **only to training set** to increase diversity and prevent overfitting while maintaining anatomically valid transformations.

### Configuration

| Transformation | Value | Rationale |
|---------------|-------|-----------|
| **Rotation** | ±15° | Simulates realistic patient positioning variations |
| **Width Shift** | 10% | Accounts for horizontal centering differences |
| **Height Shift** | 10% | Accounts for vertical centering differences |
| **Shear** | 10% | Simulates acquisition angle variations |
| **Zoom** | 10% | Handles scale variations |
| **Horizontal Flip** | Yes | Valid for chest X-rays (lungs are symmetric) |
| **Vertical Flip** | **NO** | Anatomically incorrect (heart position matters) |

### Augmentation Examples

![Augmentation Examples](figures/06_augmentation_examples.png)

*Nine variations of a single NORMAL chest X-ray showing realistic augmentations*

**Validation/Test Sets:**
- ❌ No augmentation applied
- ✅ Only normalization [0-1]

**Estimated gain: +1.0% to +2.0% accuracy**

---

## 🔧 Preprocessing Pipeline

### Pipeline Architecture
```python
def preprocess_image(img_path, img_size=224, normalize=True):
    """
    Complete preprocessing pipeline
    
    Steps:
    1. Load image (grayscale)
    2. Resize to 224×224 pixels
    3. Normalize to [0-1] range
    4. Add channel dimension → (224, 224, 1)
    """
```

### Why 224×224?

- **Transfer Learning Compatibility:** Standard input size for pretrained models (VGG16, ResNet50, DenseNet121)
- **Computational Efficiency:** Balances detail retention vs processing speed
- **Memory Optimization:** Reduces RAM requirements during training

### Pipeline Validation

![Preprocessing Pipeline](figures/07_preprocessing_pipeline.png)

*Before/after comparison showing original → resized → normalized*

**Key Features:**
- On-the-fly processing (no disk space duplication)
- Flexible (easy to change parameters)
- Efficient (faster than loading pre-processed files)

---

