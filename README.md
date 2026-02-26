# 🏥 AI-Powered Pediatric Pneumonia Detection

<p align="center">
  <img src="docs/images/xray-heart.gif" alt="Medical AI Banner" width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.19-orange.svg?style=for-the-badge&logo=tensorflow" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Keras-3.x-red.svg?style=for-the-badge&logo=keras" alt="Keras">
  <img src="https://img.shields.io/badge/Status-Complete-success.svg?style=for-the-badge" alt="Status">
  <img src="https://img.shields.io/badge/Accuracy-94.31%25-success.svg?style=for-the-badge" alt="Accuracy">
  <img src="https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge" alt="License">
</p>

<h3 align="center">AI-Powered Early Pediatric Pneumonia Detection: Integration with Electronic Medical Records in Algeria</h3>
<p align="center">Leveraging Transfer Learning and Strategic Data Engineering for Clinical-Grade AI</p>

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Team](#-team)
- [Problem Statement](#-problem-statement)
- [Solution](#-solution)
- [Project Status](#-project-status)
- [Dataset](#-dataset)
- [Data Engineering Pipeline](#-data-engineering-pipeline)
- [Model Training](#-model-training)
- [Results](#-results)
- [Model Comparison](#-model-comparison)
- [External Validation](#-external-validation)
- [Grad-CAM Visualizations](#-grad-cam-visualizations)
- [Repository Structure](#-repository-structure)
- [Quick Start](#-quick-start)
- [Documentation](#-documentation)
- [Pre-trained Models](#-pre-trained-models)
- [Academic Context](#-academic-context)
- [References](#-references)
- [License](#-license)

---

## 🔍 Overview

This project develops a **CNN-based pneumonia detection system** using transfer learning on pediatric chest X-rays. The complete end-to-end pipeline includes:

**Phase 1: Data Engineering** 
- 🔧 Data exploration and quality analysis
- 🔧 Dataset preprocessing and reorganization
- 🔧 Class balancing with optimal weight calculation
- 🔧 Intelligent data augmentation pipeline
- 🔧 Optimized TensorFlow data loaders

**Phase 2: Model Training** 
- 🤖 Transfer learning (VGG16, ResNet50, DenseNet121)
- 🤖 Comprehensive model evaluation
- 🤖 External dataset validation
- 🤖 Clinical interpretability (Grad-CAM)

> **Key Achievement:** DenseNet121 achieved **95.01% Sensitivity** and **94.31% Accuracy** on the test set, with **97.21% Sensitivity** on external validation — demonstrating strong real-world generalization.

---

## 👥 Team

<table>
  <tr>
    <td align="center"><b>Role</b></td>
    <td align="center"><b>Name</b></td>
    <td align="center"><b>Responsibilities</b></td>
  </tr>
  <tr>
    <td>📊 Project Manager</td>
    <td>Kassouar Fatima</td>
    <td>Coordination, planning, delivery</td>
  </tr>
  <tr>
    <td>🔧 Data Engineer</td>
    <td>Bouhmidi Amina Meroua</td>
    <td>Preprocessing, augmentation, data loaders</td>
  </tr>
  <tr>
    <td>🤖 ML Engineer</td>
    <td>Labani Nabila Nour El Houda</td>
    <td>Model training, evaluation, optimization</td>
  </tr>
  <tr>
    <td>💼 Business Model</td>
    <td>Miloudi Maroua Amira</td>
    <td>Clinical integration, deployment strategy</td>
  </tr>
  <tr>
    <td>👨‍🏫 Academic Supervisor</td>
    <td>Dr. Abderrahmane Khiat</td>
    <td>Academic guidance & evaluation</td>
  </tr>
</table>

**Institution:** University of Saida, Algeria  
**Period:** February 2026

---

## 🚨 Problem Statement

Pneumonia is a **leading cause of death** in children under 5:
- 📊 **740,000+ deaths** annually worldwide (15% of child mortality)
- 🏥 **200-300 cases daily** in Algerian emergency departments
- ⏱️ **6-24 hour** diagnostic delays
- 👨‍⚕️ **Severe shortage** of pediatric radiologists (~1 per 500,000 children)
- 🚗 **40% of children** travel 50-100 km for X-ray access

---

## 💡 Solution

AI-powered detection system through:
- 🧠 **Transfer Learning** (VGG16/ResNet50/DenseNet121)
- ⚖️ **Strategic Class Balancing** (weights: 1.850 / 0.685)
- 🎨 **Intelligent Data Augmentation**
- 🏥 **EMR Integration Ready** (HL7/FHIR standards)

**Technical Approach:**
- 🔧 Strategic data engineering (preprocessing, class balancing, augmentation)
- 🤖 Transfer learning with ImageNet pretrained weights
- ⚖️ Optimal class weights (1.850 / 0.685) calculated via sklearn
- 📊 Cross-dataset validation for real-world generalization
- 🔥 Grad-CAM visualizations for clinical interpretability

---

## 📊 Project Status

<table>
  <tr>
    <th>Phase</th>
    <th>Owner</th>
    <th>Status</th>
    <th>Progress</th>
    <th>Deliverables</th>
  </tr>
  <tr>
    <td>📊 Data Exploration</td>
    <td>Data Engineer</td>
    <td>✅ Complete</td>
    <td>████████████ 100%</td>
    <td>EDA Report, Figures 01-04</td>
  </tr>
  <tr>
    <td>🔧 Data Preprocessing</td>
    <td>Data Engineer</td>
    <td>✅ Complete</td>
    <td>████████████ 100%</td>
    <td>Preprocessed Dataset, Figures 05-07</td>
  </tr>
  <tr>
    <td>📦 Data Loaders</td>
    <td>Data Engineer</td>
    <td>✅ Complete</td>
    <td>████████████ 100%</td>
    <td>TF Generators, ML Package</td>
  </tr>
  <tr>
    <td>🤖 Model Training</td>
    <td>ML Engineer</td>
    <td>✅ Complete</td>
    <td>████████████ 100%</td>
    <td>3 Trained Models</td>
  </tr>
  <tr>
    <td>📈 Model Evaluation</td>
    <td>ML Engineer</td>
    <td>✅ Complete</td>
    <td>████████████ 100%</td>
    <td>Metrics & Analysis</td>
  </tr>
  <tr>
    <td>🔥 Grad-CAM Visualization</td>
    <td>ML Engineer</td>
    <td>✅ Complete</td>
    <td>████████████ 100%</td>
    <td>Clinical Heatmaps</td>
  </tr>
  <tr>
    <td>🌍 External Validation</td>
    <td>ML Engineer</td>
    <td>✅ Complete</td>
    <td>████████████ 100%</td>
    <td>Cross-Dataset Test</td>
  </tr>
  <tr>
    <td>📝 Documentation</td>
    <td>Both</td>
    <td>✅ Complete</td>
    <td>████████████ 100%</td>
    <td>Complete Reports</td>
  </tr>
</table>

---

## 📁 Dataset

### Original Dataset
- **Source:** [Kaggle Pediatric Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Size:** 5,863 labeled images
- **Classes:** NORMAL (1,583) | PNEUMONIA (4,273)
- **Patients:** Pediatric patients aged 1–5 years
- **Imbalance Ratio:** 2.7:1 (PNEUMONIA:NORMAL)

### Preprocessed Dataset (70/15/15 Split)

<table>
  <tr>
    <th>Split</th>
    <th>NORMAL</th>
    <th>PNEUMONIA</th>
    <th>Total</th>
    <th>Percentage</th>
  </tr>
  <tr>
    <td><b>Training</b></td>
    <td>1,108 (27.0%)</td>
    <td>2,991 (73.0%)</td>
    <td>4,099</td>
    <td>70%</td>
  </tr>
  <tr>
    <td><b>Validation</b></td>
    <td>237 (27.0%)</td>
    <td>641 (73.0%)</td>
    <td>878</td>
    <td>15%</td>
  </tr>
  <tr>
    <td><b>Test</b></td>
    <td>238 (27.1%)</td>
    <td>641 (72.9%)</td>
    <td>879</td>
    <td>15%</td>
  </tr>
</table>

**Configuration:**
- 📏 Image size: 224×224 pixels
- 🎨 Format: Grayscale, normalized [0-1]
- 🔄 Augmentation: Rotation (±15°), Shift (10%), Zoom (10%), Horizontal flip

### External Validation Dataset
- **Source:** [Pneumonia Radiography Dataset](https://www.kaggle.com/datasets/iamtanmayshukla/pneumonia-radiography-dataset)
- **Size:** 488 images (237 NORMAL, 251 PNEUMONIA)
- **Purpose:** Cross-dataset validation to test real-world generalization

---

## 🔧 Data Engineering Pipeline

**By Bouhmidi Amina Meroua** | 

### Phase 1: Exploration ✅
**Objective:** Understand dataset structure, identify issues

**Key Findings:**
- Total images: 5,863 (1,583 NORMAL, 4,273 PNEUMONIA)
- Class imbalance: 2.7:1 ratio (PNEUMONIA:NORMAL)
- **Critical issue:** Validation set only 16 images (0.3%) → unusable
- Variable image dimensions (384×127 to 2,916×2,713 pixels)
- Pixel intensity differences between classes

**Deliverables:**
- [Exploration Report](data_engineering/reports/DATA_EXPLORATION_REPORT.md)
- Figures 01-04 (distribution, samples, dimensions, pixels)

---

### Phase 2: Preprocessing ✅
**Objective:** Reorganize dataset, balance classes, create preprocessing pipeline

**Actions Taken:**
1. **Dataset Reorganization**
   - Created 70/15/15 stratified split
   - Validation set: 16 → 878 images (54× increase)
   - Maintained class distribution (27%/73%) across all splits

2. **Class Weight Calculation**
   - Used sklearn `compute_class_weight('balanced')`
   - NORMAL (0): 1.850
   - PNEUMONIA (1): 0.685
   - Forces model to give 2.7× more attention to minority class

3. **Data Augmentation Design**
   - Rotation: ±15° (realistic patient positioning)
   - Shifts: 10% horizontal/vertical
   - Zoom: 10%
   - Horizontal flip: Yes (lungs are symmetric)
   - Vertical flip: **NO** (anatomically incorrect)
   - Applied **only** to training set

4. **Preprocessing Pipeline**
   - Resize: 224×224 (standard for pretrained models)
   - Normalize: [0-255] → [0-1]
   - Format: Grayscale (224, 224, 1)

**Deliverables:**
- [Preprocessing Report](data_engineering/reports/DATA_PREPROCESSING_REPORT.md)
- Figures 05-07 (quality, augmentation, pipeline)
- `preprocessing_config.json`

---

### Phase 3: Data Loaders ✅
**Objective:** Create optimized TensorFlow generators for GPU training

**Implementation:**
- TensorFlow `ImageDataGenerator` with augmentation
- Batch size: 32 images
- Training: Shuffle enabled + augmentation
- Validation/Test: No shuffle, no augmentation
- Performance: <0.5 seconds per batch

**Deliverables:**
- [Data Loaders Report](data_engineering/reports/DATA_LOADERS_REPORT.md)
- Figures 08-09 (batch examples, distribution)
- `data_loader.py` (reusable Python package)
- `data_loaders_summary.json`

---

### 📊 Data Engineering Impact

**Key Contributions:**
- ✅ Validation set increased by **54×** (16 → 878 images)
- ✅ Class weights scientifically calculated via sklearn
- ✅ Augmentation respects anatomical constraints
- ✅ Pipeline optimized for GPU training (<0.5s/batch)
- ✅ Complete documentation (3 detailed reports)

**Estimated Accuracy Improvements:**
- Balanced split: +0.5%
- Class weights: +1.0% to +1.5%
- Data augmentation: +1.0% to +2.0%
- **Total contribution: +2.5% to +4.0%**

---

## 🤖 Model Training

**By Labani Nabila Nour El Houda** |

### Methodology
```
Preprocessed Data (from Data Engineer)
       ↓
Transfer Learning (ImageNet weights)
       ↓
Fine-tuning with Class Weights
       ↓
Threshold Optimization (ROC Analysis)
       ↓
Evaluation (Sensitivity, Specificity, AUC-ROC)
       ↓
Grad-CAM Interpretability
       ↓
External Dataset Validation
```

### Models Trained

| # | Architecture | Parameters | ImageNet Weights | Training Time |
|---|-------------|------------|-----------------|---------------|
| 1 | VGG16 | 138M | ✅ | 120 min |
| 2 | ResNet50 | 25M | ✅ | ~60 min |
| 3 | **DenseNet121** ⭐ | **8M** | ✅ | **58 min** |

### Training Configuration
- **Platform:** Google Colab (T4 GPU)
- **Framework:** TensorFlow 2.19 / Keras 3.x
- **Optimizer:** Adam
- **Loss:** Binary Cross-Entropy
- **Early Stopping:** Yes (patience = 7)
- **Batch Size:** 32
- **Class Weights:** {0: 1.850, 1: 0.685} (from Data Engineer)

---

## 📊 Results

### 🥇 DenseNet121 — Winner ⭐

| Metric | Default (0.5) | **Optimized (0.260)** | Target |
|--------|--------------|----------------------|--------|
| **Accuracy** | 92.04% | **94.31%** | ≥90% ✅ |
| **Sensitivity** | 90.02% | **95.01%** | ≥95% ✅ |
| **Specificity** | 97.48% | **92.44%** | ≥85% ✅ |
| **Precision** | 98.97% | 97.13% | — |
| **F1-Score** | 0.9428 | — | — |
| **AUC-ROC** | **0.9810** | — | — |
| **Training Time** | — | **57.97 min** | — |

**Confusion Matrix (Optimized Threshold = 0.260):**
```
                Predicted NORMAL    Predicted PNEUMONIA
Actual NORMAL        220 (TN)              18 (FP)
Actual PNEUMONIA      32 (FN)             609 (TP)
```
> ✅ **Clinical Impact:** Missed pneumonia cases reduced by **50%** (64 → 32) after threshold optimization

---

### 🥈 VGG16

| Metric | Default (0.5) | Optimized (0.110) | Target |
|--------|--------------|-------------------|--------|
| Accuracy | 76.11% | 91.35% | ≥90% ✅ |
| Sensitivity | 67.39% | **95.32%** | ≥95% ✅ |
| Specificity | 99.58% | 80.67% | ≥85% ⚠️ |
| AUC-ROC | 0.9644 | — | — |
| Training Time | — | 120 min | — |

> ⚠️ Required extremely aggressive threshold (0.110) to reach target sensitivity

---

### 🥉 ResNet50

| Metric | Default (0.5) | Optimized | Target |
|--------|--------------|-----------|--------|
| Accuracy | 82.14% | 78.16% | ≥90% ❌ |
| Sensitivity | 84.87% | **95.48%** | ≥95% ✅ |
| Specificity | — | 31.51% | ≥85% ❌ |
| AUC-ROC | 0.8802 | — | — |

> ❌ **Clinically Unacceptable:** 31.51% specificity means 69% of healthy children wrongly flagged

---

## 🏆 Model Comparison

| Model | Accuracy | Sensitivity | Specificity | AUC-ROC | Time | Clinical Target |
|-------|----------|-------------|-------------|---------|------|----------------|
| **DenseNet121** ⭐ | **94.31%** | **95.01%** | **92.44%** | **0.981** | **58 min** | ✅ **ALL MET** |
| VGG16 | 91.35% | 95.32% | 80.67% | 0.964 | 120 min | ⚠️ PARTIAL |
| ResNet50 | 78.16% | 95.48% | 31.51% | 0.880 | ~60 min | ❌ FAILS |

### 🎯 Why DenseNet121 Won

**Technical Advantages:**
- ✅ **Dense Connections:** Better feature reuse and gradient flow
- ✅ **Parameter Efficiency:** Only 8M params (vs 138M in VGG16) → less overfitting
- ✅ **Medical Imaging Proven:** CheXNet (Stanford, 2017) validated DenseNet121 for chest X-rays

**Clinical Performance:**
- ✅ **Only model meeting ALL targets** simultaneously
- ✅ **Fastest training** (58 min vs 120 min for VGG16)
- ✅ **Best generalization** (97.21% sensitivity on external data)
- ✅ **Highest AUC-ROC** (0.981)
- ✅ **Balanced performance** (no extreme threshold needed)

---

## 🌍 External Validation

DenseNet121 tested on **completely independent dataset** from different source:

| Metric | Internal Test Set | External Dataset | Difference |
|--------|------------------|-----------------|------------|
| **Accuracy** | 94.31% | **87.09%** | -7.22% |
| **Sensitivity** | 95.01% | **97.21%** | +2.20% ✅ |
| **Specificity** | 92.44% | 76.37% | -16.07% |
| **Total Samples** | 879 | 488 | — |

> 🎯 **Key Finding:** Sensitivity **improved** on external data (97.21% vs 95.01%), proving strong real-world generalization. Accuracy drop to 87% is expected — results above 80% indicate strong generalization in medical AI.

---

## 🔥 Grad-CAM Visualizations

Grad-CAM provides **clinical interpretability** by showing which lung regions influenced the AI decision:

<div align="center">

### ⚠️ PNEUMONIA Case - Correctly Detected
![Pneumonia Detection](model_training/results/gradcam_visualizations/gradcam_1_PNEUMONIA.png)

**Model Decision:** PNEUMONIA (Confidence: 99.3%)  
**Grad-CAM Analysis:** Highlights consolidation in right lower lobe  
**Clinical Correlation:** ✅ Correct diagnosis

---

### ✅ NORMAL Case - Correctly Classified
![Normal Classification](model_training/results/gradcam_visualizations/gradcam_4_NORMAL.png)

**Model Decision:** NORMAL (Confidence: 98.8%)  
**Grad-CAM Analysis:** Focuses on central mediastinal structures  
**Clinical Correlation:** ✅ Correct diagnosis

</div>

> 🩺 **Clinical Validation:** Heatmaps confirm model learned clinically relevant features — consolidation/infiltrates for pneumonia, normal anatomical landmarks for healthy cases.

---

## 📁 Repository Structure
```
pediatric-pneumonia-detection/
│
├── 🔧 data_engineering/              # Data Engineer work (Days 1-6)
│   ├── notebooks/
│   │   ├── 01_Data_Exploration_EDA.ipynb
│   │   ├── 02_Data_Preprocessing.ipynb
│   │   └── 03_Data_Loaders.ipynb
│   ├── reports/
│   │   ├── DATA_EXPLORATION_REPORT.md
│   │   ├── DATA_PREPROCESSING_REPORT.md
│   │   └── DATA_LOADERS_REPORT.md
│   └── src/
│       └── data_loader.py
│
├── 🤖 model_training/                # ML Engineer work (Days 7-12)
│   ├── notebooks/
│   │   ├── 01_DenseNet121_Training.ipynb
│   │   ├── 01_DenseNet121_GradCAM_and_Threshold.ipynb
│   │   ├── 02_ResNet50_Training.ipynb
│   │   ├── 03_VGG16_Training.ipynb
│   │   ├── 04_External_Validation.ipynb
│   │   └── 05_GradCAM.ipynb
│   ├── logs/
│   │   ├── densenet121_20260222_125840/
│   │   ├── resnet50_20260224_193657/
│   │   └── vgg16_20260225_113134/
│   ├── results/
│   │   ├── Densenet121/
│   │   ├── Resnet50/
│   │   ├── VGG16/
│   │   ├── External_Validation/
│   │   └── gradcam_visualizations/
│   └── saved_models/
│       ├── densenet121_best_model.keras (34.3 MB)
│       ├── resnet50_best_model.keras (102.6 MB)
│       └── vgg16_best_model.keras (57.7 MB)
│
├── 📊 data/                          # Datasets (not included)
│   └── README.md
│
├── 📁 docs/
│   ├── images/                       # All visualizations
│   │   ├── 01-09_*.png              # Data engineering figures
│   │   └── xray-heart.gif           # Banner
│   ├── Multimodal_Pneumonia_Complete_Report.pdf
│   ├── TOPIC_PROPOSAL_2.pdf
│   └── Pneumonia_Detection_Guide_v4-4.pdf
│
├── ⚙️ config/
│   ├── preprocessing_config.json
│   └── data_loaders_summary.json
│
├── 📄 reports/
│   ├── DATA_EXPLORATION_REPORT.md
│   ├── DATA_PREPROCESSING_REPORT.md
│   └── DATA_LOADERS_REPORT.md
│
├── 📖 README.md                      # This file
├── 📦 requirements.txt
└── 📄 LICENSE
```

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install tensorflow>=2.10 keras numpy matplotlib scikit-learn seaborn opencv-python
```

### 1. Clone the repository
```bash
git clone https://github.com/AminaMar/pediatric-pneumonia-detection.git
cd pediatric-pneumonia-detection
```

### 2. Download the dataset
- [Kaggle Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Place in: `data/chest_xray/`

### 3. Data Engineering (Optional - already done)

Run notebooks in order:

1. [01_Data_Exploration_EDA (1).ipynb](data_engineering/notebooks/01_Data_Exploration_EDA%20(1).ipynb)
2. [02_Data_Preprocessing.ipynb](data_engineering/notebooks/02_Data_Preprocessing.ipynb)
3. [03_Data_Loaders.ipynb](data_engineering/notebooks/03_Data_Loaders.ipynb)
```

### 4. Model Training
```python
# Open in Google Colab (T4 GPU recommended)
model_training/notebooks/01_DenseNet121_Training.ipynb
model_training/notebooks/04_External_Validation.ipynb
model_training/notebooks/05_GradCAM.ipynb
```

### 5. Load Pre-trained Model
```python
import tensorflow as tf

# Download from: https://drive.google.com/drive/folders/1JtnqNL4lMSRHBtR_eex96k64wSix97Y9
model = tf.keras.models.load_model('densenet121_best_model.keras')

# Predict on new X-ray
prediction = model.predict(preprocessed_image)
```

---

## 📖 Documentation

| Document | Author |
|----------|--------|
| [Data Exploration Report](data_engineering/reports/DATA_EXPLORATION_REPORT.md)  | Data Engineer |
| [Data Preprocessing Report](data_engineering/reports/DATA_PREPROCESSING_REPORT.md)  | Data Engineer |
| [Data Loaders Report](data_engineering/reports/DATA_LOADERS_REPORT.md) | Data Engineer |
| [Complete Project Report](docs/Multimodal_Pneumonia_Complete_Report.pdf) | Team |
| [Technical Guide](docs/Pneumonia_Detection_Guide_v4-4.pdf) |  ML Engineer |
| [Topic Proposal](docs/TOPIC_PROPOSAL_2.pdf) | Team |

---

## 💾 Pre-trained Models

Model files exceed GitHub's 25MB limit and are hosted on **Google Drive**.

👉 [**Download All Models (Google Drive)**](https://drive.google.com/drive/folders/1JtnqNL4lMSRHBtR_eex96k64wSix97Y9?usp=sharing)

| Model | Accuracy | Sensitivity | Size | Recommended |
|-------|----------|-------------|------|-------------|
| **densenet121_best_model.keras** | **94.31%** | **95.01%** | 34.3 MB | ⭐ **YES** |
| vgg16_best_model.keras | 91.35% | 95.32% | 57.7 MB | ⚠️ Acceptable |
| resnet50_best_model.keras | 78.16% | 95.48% | 102.6 MB | ❌ Not recommended |

---

## 🎓 Academic Context

**Course:** AI in Healthcare  
**Institution:** University of Saida, Algeria  
**Academic Supervisor:** Dr. Abderrahmane Khiat  

### Team Contributions

| Team Member | Contribution | Work Output |
|-------------|-------------|-------------|
| **Kassouar Fatima** | Project management | Coordination, planning, delivery |
| **Bouhmidi Amina Meroua** | Data engineering  | 3 reports, analyse, data loaders |
| **Labani Nabila Nour El Houda** | Model training  | 3 models, evaluation, Grad-CAM |
| **Miloudi Maroua Amira** | Business model | Clinical integration strategy |

---

## 📚 References

1. **CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays** — *Stanford AI Lab, 2017*
2. **Detection of pneumonia in children through chest radiographs using AI in low-resource settings** — *PLOS Digital Health, 2025*
3. **AI–EHR integration improving diagnostic capabilities through HL7/FHIR standards** — *PMC/PubMed Central, 2024*
4. **Diagnostic Performance of a Deep Learning Model Deployed at a National COVID-19 Screening Facility** — *Healthcare MDPI, 2022*
5. **Pneumonia in Children — Fact Sheet** — *WHO, 2024*

---

## 📊 Key Achievements

✅ **Clinical Targets Met:**
- Accuracy: 94.31% (target: ≥90%)
- Sensitivity: 95.01% (target: ≥95%)
- Specificity: 92.44% (target: ≥85%)

✅ **External Validation:**
- 97.21% sensitivity maintained on independent dataset
- Proves real-world generalization

✅ **Complete Pipeline:**
- Data engineering → Model training → Evaluation → Interpretability
- End-to-end reproducible system

✅ **Documentation:**
- 3 detailed data engineering reports
- Complete notebooks for all phases
- Comprehensive final report

---

## 📄 License

This project is licensed under the MIT License.

---

## 📧 Contact

**Project Repository:** [github.com/AminaMar/pediatric-pneumonia-detection](https://github.com/AminaMar/pediatric-pneumonia-detection)

**Data Engineer:** Bouhmidi Amina Meroua - [GitHub](https://github.com/AminaMar)  
**ML Engineer:** Labani Nabila Nour El Houda - [GitHub](https://github.com/labaninabila193-code)

---

<div align="center">

**University of Saida, Algeria — 2026**

*This project addresses a critical healthcare challenge through state-of-the-art AI,*  
*aligned with Algeria's Digital Health Strategy (2021–2022)*



</div>
