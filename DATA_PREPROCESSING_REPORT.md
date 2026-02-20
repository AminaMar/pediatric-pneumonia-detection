📊 Data Preprocessing & Augmentation – Executive Summary

Data Engineer: Bouhmidi Amina Maroua
Date: February 20, 2026
Project: AI-Powered Early Pediatric Pneumonia Detection: Integration with Electronic Medical Records in Algeria
Phase: Data Preprocessing & Augmentation

1️⃣ Dataset Reorganization

Original Issues:

Tiny validation set (16 images)

Uneven class distribution

No quality filtering

Solution: Stratified 70/15/15 split

Split	NORMAL	PNEUMONIA	Total	%
Train	1,108	2,991	4,099	70%
Validation	237	641	878	15%
Test	238	641	879	15%
Total	1,583	4,273	5,856	100%

✅ Balanced splits and unbiased evaluation
Estimated gain: +0.5% accuracy

2️⃣ Image Quality Analysis

Metrics Checked:

Sharpness (Laplacian variance)

Brightness (mean pixel intensity)

Contrast (standard deviation)

Decision: All images retained; optional filtering improves +0.5% accuracy

Example:

![Quality Analysis](figures/05_quality_analysis.png)


3️⃣ Class Imbalance Handling

Dataset Imbalance: 2.7:1 (PNEUMONIA:NORMAL)

Rejected Approaches:

Undersampling → data loss

Oversampling → risk of overfitting

SMOTE → may produce invalid images

Implemented:

Stratified split

Class weights: NORMAL=1.850, PNEUMONIA=0.685

Targeted augmentation of NORMAL

Impact: Forces equal learning, estimated +1.0–1.5% accuracy

4️⃣ Data Augmentation

Applied only to training set

Transformation	Value
Rotation	±15°
Width/Height Shift	10%
Shear	10%
Zoom	10%
Horizontal Flip	Yes
Vertical Flip	No

Example Augmentations:
![Augmentation Examples](figures/06_augmentation_examples.png)


Estimated gain: +1.0–2.0% accuracy

5️⃣ Preprocessing Pipeline

Steps:

Load grayscale image

Resize → 224×224 px

Normalize [0–1]

Add channel dimension → (224, 224, 1)

Rationale:

Compatible with pretrained models (VGG16, ResNet50)

Efficient memory & computation

On-the-fly processing

Pipeline Visualization:
![Preprocessing Pipeline](figures/07_preprocessing_pipeline.png)


6️⃣ Expected Improvements
Step	Gain
Balanced split	+0.5%
Class weights	+1.0–1.5%
Data augmentation	+1.0–2.0%
Quality filtering	+0.5%
Total	+3.0–4.5%
Baseline	96.4%
Target	97.0%+
7️⃣ Deliverables

Preprocessed dataset: data/preprocessed/ ✅

JSON config: preprocessing_config.json ✅

Figures: figures/05–07_*.png ✅

Notebook: 02_Data_Preprocessing.ipynb ✅

Summary JSON: preprocessing_summary.json ✅

Status: ✅ Complete, ready for model training (Days 5–6)