
# AI-Powered Pediatric Pneumonia Detection

<p align="center">
  <img src="docs/images/xray-heart.gif" alt="Medical AI Banner" width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.19-orange.svg?style=for-the-badge&logo=tensorflow" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Keras-3.x-red.svg?style=for-the-badge&logo=keras" alt="Keras">
  <img src="https://img.shields.io/badge/Streamlit-Latest-FF4B4B.svg?style=for-the-badge&logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/Status-Complete-success.svg?style=for-the-badge" alt="Status">
  <img src="https://img.shields.io/badge/Sensitivity-97.21%25-success.svg?style=for-the-badge" alt="Sensitivity">
  <img src="https://img.shields.io/badge/Accuracy-94.31%25-success.svg?style=for-the-badge" alt="Accuracy">
  <img src="https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge" alt="License">
</p>

<h3 align="center">AI-Powered Early Pediatric Pneumonia Detection: Integration with Electronic Medical Records in Algeria</h3>
<p align="center">Leveraging Transfer Learning, Strategic Data Engineering, and Full-Stack Clinical Application Development</p>

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
- [Clinical Application](#-clinical-application)
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

This project delivers a **complete, end-to-end AI-powered clinical decision support system** for pediatric pneumonia detection, from raw data engineering through to a fully deployed Streamlit application used by clinicians in real time.

The pipeline spans four major pillars:

**Pillar 1 — Data Engineering** *(Bouhmidi Amina Meroua)*
- Complete EDA on 5,863 pediatric chest X-rays
- Dataset reorganization with stratified 70/15/15 split
- Scientifically computed class weights and intelligent augmentation
- Optimized TensorFlow data loaders for GPU training

**Pillar 2 — Model Training & Deep Learning** *(Labani Nabila Nour El Houda)*
- Transfer learning pipeline across VGG16, ResNet50, and DenseNet121
- Threshold optimization via ROC analysis (0.5 → 0.260)
- GradCAM clinical interpretability implementation
- External cross-dataset validation achieving **97.21% sensitivity**

**Pillar 3 — RAG-Based Generative AI, Clinical Application Development & Automation + Business Model** *(Miloudi Maroua Amira — Developer & Business Model)*
- Designed and implemented a RAG-based clinical reasoning system (FAISS + BM25 + reranking) for patient-specific medical inference
- Engineered the LLM orchestration pipeline via OpenRouter with structured clinical output (RISK / REASONING / ACTION)
- Developed the prompt engineering framework with embedded clinical heuristics for reliable AI-assisted decision support
- Built and integrated the full-stack Streamlit clinical application, connecting AI models, agent modules, and user workflows
- Implemented automation pipelines including intelligent reasoning workflows and automated medical PDF report generation
- Designed the zero-infrastructure file-based system architecture enabling deployment in low-resource clinical environments
- Defined the business model and deployment strategy for scalable, cost-effective adoption

**Pillar 4 — Vital Signs ML Pipeline** *(Kassouar Fatima)*
- Gradient Boosting classifier for vital signs–based diagnosis
- CSV preprocessing pipeline and data management
- Binary probability scoring ∈ [0,1] per patient record (12 clinical features)
- Feature engineering and data cleaning

> **Key Achievement:** DenseNet121 achieved **95.01% Sensitivity** and **94.31% Accuracy** on the internal test set, improving to **97.21% Sensitivity** on external validation. The full clinical application integrates both the deep learning model and a Gradient Boosting vital signs classifier into a production-grade Streamlit interface.

---

## 👥 Team

| Role | Name | Core Responsibilities |
|------|------|-----------------------|
| 🔧 **Data Engineer** | **Bouhmidi Amina Meroua** | Complete data pipeline (EDA → preprocessing → data loaders), 70/15/15 stratified split, class weight computation, augmentation design |
| 🤖 **Deep Learning Engineer** | **Labani Nabila Nour El Houda** | DenseNet121 / VGG16 / ResNet50 training, threshold optimization (0.260), GradCAM implementation, external dataset validation (97.21% sensitivity), X-ray diagnostic module in app |
| 💻 **Developer & Business Model** | **Miloudi Maroua Amira** | Built an AI-powered medical app with Streamlit, featuring intelligent data explanations, workflow automation, reporting, and real-world deployment planning |
| 📋 **Manager** | **Kassouar Fatima** | Gradient Boosting classifier (vital signs), CSV preprocessing pipeline and management |
| 👨‍🏫 Academic Supervisor | Dr. Abderrahmane Khiat | Academic guidance & evaluation |
| 🏥 Medical Advisor | Dr. Aimer Mohammed Djamel Eddine | Clinical validation, medical references & advisory |

**Institution:** University of Saida, Algeria — Academic Year 2025–2026

---

## 🚨 Problem Statement

Pneumonia is a **leading cause of death** in children under 5:

- 📊 **740,000+ deaths** annually worldwide (15% of child mortality)
- 🏥 **200–300 cases daily** in Algerian emergency departments
- ⏱️ **6–24 hour** diagnostic delays due to radiologist shortage
- 👨‍⚕️ **~1 pediatric radiologist per 500,000 children** — a 500× deficiency vs. developed nations
- 🚗 **40% of children** must travel 50–100 km to access chest X-ray imaging
- 📉 **70–85% radiologist agreement** on pneumonia diagnosis — significant inter-observer error

---

## 💡 Solution

A dual-pathway AI system with a full clinical application:

- 🧠 **Deep Learning** — DenseNet121 fine-tuned on 5,856 pediatric chest X-rays (95.01% sensitivity)
- 🌿 **Machine Learning** — Gradient Boosting classifier on 12 vital sign parameters
- 🔥 **GradCAM Explainability** — Clinical heatmaps showing which lung regions influenced each decision
- ⚖️ **Strategic Class Balancing** — Computed weights (1.850 / 0.685) via sklearn
- 🎨 **Intelligent Data Augmentation** — Anatomically-constrained (no vertical flips)
- 🤖 **RAG-Based Generative AI** — Clinical reasoning chatbot combining patient data and medical literature (FAISS + BM25 + reranking) for evidence-based decision support
- ⚙️ **Clinical Automation System** — Automated workflows for reasoning, diagnosis support, and patient data processing
- 📄 **Automated Medical PDF Reports** — Generation of structured patient reports including contact information, medical case, diagnostics, and treatment plans
- 🗂️ **Patient Data Persistence** — Complete file-based system storing patient records, history, diagnostics, treatments, and documents for long-term tracking
- 🏥 **EMR Integration Ready** — HL7/FHIR compatible architecture
- 🖥️ **Full Clinical Application** — Streamlit app: patient registration, dual AI diagnostics, treatment planning, nurse monitoring
- 🚑 **Emergency Protocol System** — Doctor-to-emergency-doctor communication workflow for critical cases
- 👨‍👩‍👧 **Parental Guidance Module** — Clear instructions and safety guidance for families in severe or emergency situations
- 🗄️ **Zero-Infrastructure Deployment** — File-based persistence, no database server required

---

## 📊 Project Status

| Phase | Lead | Status | Progress | Deliverables |
|-------|------|--------|----------|--------------|
| 📊 Data Pipeline (Exploration, Preprocessing, Loaders) | Bouhmidi Amina Meroua | ✅ Complete | 100% | EDA Report, Figures 01–07, Preprocessed Dataset, preprocessing_config.json, TF Data Generators, data_loader.py |
| 🤖 Deep Learning Pipeline (Training, Evaluation, Interpretability, Validation) | Labani Nabila Nour El Houda | ✅ Complete | 100% | VGG16, ResNet50, DenseNet121, ROC Analysis, Metrics Report, Grad-CAM Heatmaps, External Validation (97.21% sensitivity) |
| 🖥️ RAG + Clinical Streamlit Application | Miloudi Maroua Amira | ✅ Complete | 100% | Doctor onboarding, patient management, AI diagnostics |
| 🌿 Vital Signs ML Module | Kassouar Fatima | ✅ Complete | 100% | Gradient Boosting, CSV pipeline, feature engineering |
| 📝 Reports & Representation | Full Team | ✅ Complete | 100% | Final report, technical guide, presentations |

---

## 📁 Dataset

### Original Dataset

- **Source:** [Kaggle Pediatric Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Size:** 5,863 labeled images
- **Classes:** NORMAL (1,583) | PNEUMONIA (4,273)
- **Patients:** Pediatric patients aged 1–5 years
- **Imbalance Ratio:** 2.7:1 (PNEUMONIA:NORMAL)

### Preprocessed Dataset (70/15/15 Stratified Split)

| Split | NORMAL | PNEUMONIA | Total | Percentage |
|-------|--------|-----------|-------|------------|
| **Training** | 1,108 (27.0%) | 2,991 (73.0%) | 4,099 | 70% |
| **Validation** | 237 (27.0%) | 641 (73.0%) | 878 | 15% |
| **Test** | 238 (27.1%) | 641 (72.9%) | 879 | 15% |

**Configuration:**
- 📏 Image size: 224×224 pixels
- 🎨 Format: Grayscale, normalized [0-1]
- 🔄 Augmentation: Rotation (±15°), Shift (10%), Zoom (10%), Horizontal flip only

### External Validation Dataset

- **Source:** [Pneumonia Radiography Dataset](https://www.kaggle.com/datasets/iamtanmayshukla/pneumonia-radiography-dataset)
- **Size:** 488 images (237 NORMAL, 251 PNEUMONIA)
- **Purpose:** Cross-dataset validation to test real-world generalization

---

## 🔧 Data Engineering Pipeline

**Lead: Bouhmidi Amina Meroua**

### Phase 1 — Data Exploration ✅

**Objective:** Understand dataset structure, identify quality issues

**Key Findings:**
- Total images: 5,863 (1,583 NORMAL, 4,273 PNEUMONIA)
- Class imbalance: 2.7:1 ratio (PNEUMONIA:NORMAL)
- **Critical issue detected:** Original validation set only 16 images (0.3%) → statistically unusable
- Variable image dimensions: 384×127 px to 2,916×2,713 px
- Measurable pixel intensity differences between classes

**Deliverables:**
- [Exploration Report](data_engineering/reports/DATA_EXPLORATION_REPORT.md)
- Figures 01–04: distribution analysis, sample review, dimension mapping, pixel statistics

---

### Phase 2 — Preprocessing ✅

**Objective:** Reorganize dataset, balance classes, build preprocessing pipeline

**Actions Taken:**

**1. Dataset Reorganization**
- Created 70/15/15 stratified split
- Validation set: 16 → 878 images (**54× increase**)
- Class distribution (27%/73%) maintained across all three splits

**2. Class Weight Calculation**
- Method: `sklearn.utils.class_weight.compute_class_weight('balanced')`
- NORMAL (class 0): **1.850** — forces model to penalize missed normals
- PNEUMONIA (class 1): **0.685**
- Result: model gives 2.7× more learning attention to the minority class

**3. Augmentation Strategy** (anatomically constrained)
- Rotation: ±15° (realistic patient positioning variation)
- Horizontal/vertical shifts: 10%
- Zoom: 10%
- Horizontal flip: ✅ (lungs are symmetric)
- Vertical flip: ❌ (anatomically incorrect — never applied)
- Applied **only to training set** — validation/test are untouched

**4. Preprocessing Pipeline**
- Resize: 224×224 (required by ImageNet pretrained architectures)
- Normalize: [0–255] → [0–1] float32
- Format: Grayscale (224, 224, 1)

**Deliverables:**
- [Preprocessing Report](data_engineering/reports/DATA_PREPROCESSING_REPORT.md)
- Figures 05–07: quality checks, augmentation samples, pipeline diagram
- `preprocessing_config.json`

---

### Phase 3 — Data Loaders ✅

**Objective:** Build optimized TensorFlow generators for GPU training

**Implementation:**
- `tf.keras.preprocessing.image.ImageDataGenerator` with full augmentation pipeline
- Batch size: 32 images
- Training: shuffle enabled + augmentation active
- Validation / Test: no shuffle, no augmentation (clean evaluation)
- **Performance: <0.5 seconds per batch** on T4 GPU

**Deliverables:**
- [Data Loaders Report](data_engineering/reports/DATA_LOADERS_REPORT.md)
- Figures 08–09: batch sample visualization, class distribution per split
- `data_loader.py` — reusable, importable Python package
- `data_loaders_summary.json`

---

### 📊 Data Engineering Impact

| Contribution | Before | After | Impact |
|---|---|---|---|
| Validation set size | 16 images | 878 images | **+54× increase** |
| Class weight handling | None | 1.850 / 0.685 | Minority class emphasis |
| Augmentation | None | 5 transforms | +1–2% accuracy |
| Preprocessing pipeline | Raw pixels | 224×224 normalized | Training-ready format |
| Documentation | None | 3 detailed reports | Full reproducibility |

**Estimated accuracy contribution from data engineering alone: +2.5% to +4.0%**

---

## 🤖 Model Training

**Lead: Labani Nabila Nour El Houda** ([@labaninabila193-code](https://github.com/labaninabila193-code))

### Methodology

```
Preprocessed Data (from Data Engineer)
       ↓
Transfer Learning (ImageNet pretrained weights)
       ↓
Fine-tuning with Computed Class Weights
       ↓
Threshold Optimization via ROC Analysis (0.5 → 0.260)
       ↓
Evaluation (Sensitivity, Specificity, AUC-ROC, F1)
       ↓
GradCAM Clinical Interpretability
       ↓
External Dataset Validation (488 independent images)
```

### Models Trained

| # | Architecture | Parameters | ImageNet Weights | Training Time |
|---|---|---|---|---|
| 1 | VGG16 | 138M | ✅ | 120 min |
| 2 | ResNet50 | 25M | ✅ | ~60 min |
| 3 | **DenseNet121** ⭐ | **8M** | ✅ | **58 min** |

### Training Configuration

- **Platform:** Google Colab (T4 GPU)
- **Framework:** TensorFlow 2.19 / Keras 3.x
- **Optimizer:** Adam
- **Loss:** Binary Cross-Entropy
- **Early Stopping:** Patience = 7 epochs
- **Batch Size:** 32
- **Class Weights:** {0: 1.850, 1: 0.685} — provided by Data Engineer

---

## 🌿 Vital Signs ML Module

**Lead: Kassouar Fatima**

Kassouar Fatima developed the machine learning pipeline for vital signs-based pneumonia classification, building the Gradient Boosting model and the complete medical explanation engine integrated into the clinical application.

### Gradient Boosting Classifier

| Attribute | Detail |
|---|---|
| **File** | `models/Gradient_Boost.pkl` |
| **Framework** | scikit-learn 1.6.1 (serialized via joblib) |
| **Input** | 12-feature encoded vector (categoricals + numerics) |
| **Output** | Binary class (0 = Not Sick, 1 = Sick) + probability score |
| **Top feature** | Confusion — **64.8% importance** |

### Feature Importance Ranking

| Rank | Feature | Importance |
|---|---|---|
| 1 | Confusion | **64.8%** |
| 2 | Fever severity | 12.9% |
| 3 | Temperature | 9.2% |
| 4–12 | Remaining 9 features | 13.1% combined |

### Input Features (12 Parameters)

| # | Feature | Type |
|---|---|---|
| 1 | Gender | Categorical |
| 2 | Age (1–16) | Integer |
| 3 | Cough type | Categorical |
| 4 | Fever severity | Categorical |
| 5 | Shortness of breath | Boolean |
| 6 | Chest pain | Boolean |
| 7 | Fatigue | Boolean |
| 8 | **Confusion** | Boolean (top feature: 64.8%) |
| 9 | Oxygen saturation (SpO₂) | Float |
| 10 | Crackles | Boolean |
| 11 | Sputum color | Categorical |
| 12 | Temperature | Float |

---

## 🖥️ RAG & Automation + Clinical Application

**Developer & Business Model: Miloudi Maroua Amira** ([@lily01011](https://github.com/lily01011))

The clinical application was fully designed and engineered by Miloudi Maroua Amira, transforming standalone AI models into a complete intelligent clinical decision support system. The system integrates machine learning, deep learning, and a Retrieval-Augmented Generation (RAG) pipeline into a production-ready application deployable in any clinical environment with zero infrastructure requirements.

Beyond system development, the business model and deployment strategy were also defined by Miloudi Maroua Amira, ensuring real-world usability in low-resource healthcare settings.

### Dual AI Diagnostic Pathways

| Pathway | Input | Model | Output |
|---|---|---|---|
| **Chest X-Ray Diagnosis** | Pediatric chest radiograph (PNG/JPG) | DenseNet121 CNN | PNEUMONIA / NORMAL + GradCAM heatmap + radiological explanation |
| **Vital Signs Diagnosis** | 12 clinical parameters | Gradient Boosting Classifier | Sick / Not Sick + risk score (0–100) + per-feature medical explanation |

### 🧠 Intelligent RAG & Clinical Reasoning Layer (Agent System)

RAG-powered clinical chatbot combining:
- Patient-specific documents (PDF ingestion)
- Medical literature knowledge base (FAISS-indexed)
- Hybrid retrieval:
  - Semantic search (FAISS embeddings)
  - Keyword search (BM25)
  - Cross-encoder reranking (Cohere / FlashRank / fallback)
- Structured clinical reasoning via prompt engineering:
  - Output format: RISK / REASONING / ACTION
  - Embedded clinical inference rules (treatment failure, escalation detection, emergency patterns)
- Real-time LLM response streaming via OpenRouter API
- Session-based knowledge base with conversational memory
- Enables doctor-level decision support, not just prediction

### 🤖 Automation & Clinical Workflow Intelligence

End-to-end automated clinical workflow:
- Patient registration → diagnosis → treatment → monitoring → reporting
- Intelligent data aggregation across all modules: diagnostics history, treatments, nurse monitoring tables, uploaded documents
- Automated reasoning over temporal clinical data trends (not single snapshots)
- Reduces manual workload while improving clinical consistency

### 📄 Automated Medical Report Generation

Full A4 clinical PDF report generation using ReportLab. Automatically compiles:
- Patient contact information
- Complete medical case history
- Diagnostic results (vital + X-ray)
- Treatment plans (home + hospital)
- Nurse monitoring timeline
- Uploaded medical documents
- Colour-coded clinical tables for readability
- Timestamped reports for audit trail (`agent_report_<timestamp>.pdf`)
- Supports clinical handover, documentation, and archiving

### 🗄️ Patient Data Management & Persistence

Full file-based EMR-like system (no database required):
- Structured CSV records for all clinical data
- Organized patient folders with complete history
- Stores: diagnostics (append-only history), treatments and hospitalisation records, nurse monitoring intervals, X-ray images, uploaded PDFs
- Ensures: data traceability, historical integrity, doctor-level data isolation
- Enables deployment in low-resource environments without infrastructure

### 👩‍⚕️ Enhanced Clinical User Experience

**Emergency Treatment Protocol**
- Structured doctor-to-emergency-doctor handover logic
- Critical escalation pathways based on patient condition trends

**Parental Guidance System**
- Clear instructions for caregivers in severe cases
- Emergency warning signs and actionable steps
- Designed for fast clinical decision-making, reduced cognitive load on physicians, and real-world usability in high-pressure scenarios

### Data Integrity Mechanisms

- **Session gating:** All operations verify active doctor session before any file I/O
- **Append-only diagnostics:** Sequential `N-vitaldiagnostic.csv` naming — immutable history
- **Two-step save workflow:** AI results displayed for doctor review before any file is written
- **Path sanitization:** All folder names sanitized; no OS-reserved characters
- **UTF-8 enforcement:** Correct handling of Arabic-French patient names
- **Zero-infrastructure:** Runs on any machine with Python — no network, no database server

### System Architecture

```
╔══════════════════════════════════════════════════════════════════╗
║                   TIER 1 — PRESENTATION LAYER                   ║
║  app.py (Home & Nav Hub)  ·  profile.py  ·  Add_Patient.py      ║
║  vitaldiagnostique.py  ·  xraydiagnostique.py                   ║
║  Treatment.py  ·  Hospitaltreatment.py  ·  About_Us.py          ║
╚══════════════════════════════╦═══════════════════════════════════╝
                               ║ function calls only
╔══════════════════════════════╩═══════════════════════════════════╗
║                TIER 2 — BUSINESS LOGIC & AI LAYER               ║
║  profile_db.py       ← Session management & auth boundary       ║
║  patient_db.py       ← Patient CRUD + PDF/CSV generation        ║
║  vitaldiagnostique_db.py  ← Append-only diagnostic records      ║
║  xraydiagnostique_db.py   ← X-ray image storage                 ║
║  treatment_db.py     ← Treatment plans + hospital monitoring    ║
║  whyvitals.py        ← Vital signs medical explanation engine   ║
║  whyxray.py          ← Radiology explanation dictionary         ║
║  Gradient_Boost.pkl  ← GBM inference (scikit-learn)             ║
║  DenseNet121.keras   ← CNN inference + GradCAM (TensorFlow)     ║
╚══════════════════════════════╦═══════════════════════════════════╝
                               ║ imported by Streamlit UI
╔══════════════════════════════╩═══════════════════════════════════╗
║              TIER 3 — INTELLIGENT AGENT LAYER (NEW v9)          ║
║  agentt/layer1.py      ← PDF ingestion → FAISS + BM25 index     ║
║  agentt/retrieval.py   ← Hybrid cross-KB search + reranking     ║
║  agentt/prompt_builder.py ← Structured clinical prompt assembly ║
║  agentt/layer2.py      ← LLM orchestration + streaming (OpenRouter) ║
║  agentt/report_agent.py ← Patient PDF report generation         ║
║  agentt/med_kb.py      ← Medical literature KB + FAISS cache    ║
║  agentt/agent_utils.py ← Shared data-access utilities           ║
╚══════════════════════════════╦═══════════════════════════════════╝
                               ║ pathlib.Path file I/O
╔══════════════════════════════╩═══════════════════════════════════╗
║                  TIER 4 — PERSISTENCE LAYER                     ║
║  data/<Region-Hospital>/<DoctorName>/<PatientID_Name>/          ║
║  doctor_profile.csv  ·  N-vitaldiagnostic.csv                   ║
║  home_treatment.csv  ·  hospitalizedtreatment.csv               ║
║  Check Table N.csv   ·  xray/*.png  ·  *_pdf/ folders           ║
║  agent_report_<timestamp>.pdf  ← Generated by report_agent.py   ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 📊 Results

### 🥇 DenseNet121 — Best Model ⭐

| Metric | Default Threshold (0.5) | **Optimized Threshold (0.260)** | Clinical Target |
|---|---|---|---|
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
Actual NORMAL            220 (TN)              18 (FP)
Actual PNEUMONIA          32 (FN)             609 (TP)
```

> ✅ **Clinical Impact:** Missed pneumonia cases reduced by **50%** (64 → 32) after threshold optimization.

---

### 🥈 VGG16

| Metric | Default (0.5) | Optimized (0.110) | Target |
|---|---|---|---|
| Accuracy | 76.11% | 91.35% | ≥90% ✅ |
| Sensitivity | 67.39% | 95.32% | ≥95% ✅ |
| Specificity | 99.58% | 80.67% | ≥85% ⚠️ |
| AUC-ROC | 0.9644 | — | — |

> ⚠️ Required an extreme threshold (0.110) to reach sensitivity target. Specificity remains below clinical threshold.

---

### 🥉 ResNet50

| Metric | Default (0.5) | Optimized | Target |
|---|---|---|---|
| Accuracy | 82.14% | 78.16% | ≥90% ❌ |
| Sensitivity | 84.87% | 95.48% | ≥95% ✅ |
| Specificity | — | 31.51% | ≥85% ❌ |
| AUC-ROC | 0.8802 | — | — |

> ❌ **Clinically unacceptable:** 31.51% specificity means 69% of healthy children are incorrectly flagged as pneumonia cases.

---

## 🏆 Model Comparison

| Model | Accuracy | Sensitivity | Specificity | AUC-ROC | Time | Clinical Verdict |
|---|---|---|---|---|---|---|
| **DenseNet121** ⭐ | **94.31%** | **95.01%** | **92.44%** | **0.981** | **58 min** | ✅ **ALL TARGETS MET** |
| VGG16 | 91.35% | 95.32% | 80.67% | 0.964 | 120 min | ⚠️ Partial |
| ResNet50 | 78.16% | 95.48% | 31.51% | 0.880 | ~60 min | ❌ Not suitable |

### Why DenseNet121 Won

- ✅ **Only model meeting ALL clinical targets simultaneously**
- ✅ **Dense connections** — superior feature reuse, better gradient flow
- ✅ **Parameter efficiency** — 8M vs. 138M (VGG16) → less overfitting risk
- ✅ **Clinically validated architecture** — Stanford CheXNet (2017) used DenseNet121 for chest X-rays
- ✅ **Fastest training** — 58 min vs. 120 min for VGG16
- ✅ **Best AUC-ROC** — 0.981
- ✅ **Balanced performance** — no extreme threshold manipulation needed

---

## 🌍 External Validation

DenseNet121 tested on a **completely independent dataset** from a different source (488 images, never seen during training or validation):

| Metric | Internal Test Set | External Dataset | Difference |
|---|---|---|---|
| **Accuracy** | 94.31% | 87.09% | -7.22% |
| **Sensitivity** | 95.01% | **97.21%** | **+2.20%** ✅ |
| **Specificity** | 92.44% | 76.37% | -16.07% |
| Total Samples | 879 | 488 | — |

> 🎯 **Key Finding:** Sensitivity **improved** on the external dataset (97.21% vs. 95.01%), confirming the model generalizes to real-world, unseen clinical data. Accuracy above 80% on a fully independent dataset is considered strong generalization in medical AI literature.

---

## 🔥 Grad-CAM Visualizations

Grad-CAM highlights which lung regions influenced each AI decision, enabling clinical verification and building radiologist trust in the system:

<div align="center">

### ⚠️ PNEUMONIA Case — Correctly Detected

![Pneumonia Detection](model_training/results/gradcam_visualizations/gradcam_1_PNEUMONIA.png)

**Model Decision:** PNEUMONIA (Confidence: 99.3%)
**Grad-CAM:** Highlights consolidation in right lower lobe
**Clinical Correlation:** ✅ Correct — matches radiological findings

---

### ✅ NORMAL Case — Correctly Classified

![Normal Classification](model_training/results/gradcam_visualizations/gradcam_4_NORMAL.png)

**Model Decision:** NORMAL (Confidence: 98.8%)
**Grad-CAM:** Focuses on central mediastinal structures
**Clinical Correlation:** ✅ Correct — no pathological features highlighted

</div>

> 🩺 **Clinical Validation:** Heatmaps confirm the model learned clinically meaningful features — consolidation/infiltrates for pneumonia cases, normal anatomical landmarks for healthy cases — rather than dataset artifacts.

---

## 📁 Repository Structure

```
pediatric-pneumonia-detection/
│
├── 🔧 data_engineering/               # Bouhmidi Amina Meroua — Phases 1–3
│   ├── notebooks/
│   │   ├── 01_Data_Exploration_EDA.ipynb
│   │   ├── 02_Data_Preprocessing.ipynb
│   │   └── 03_Data_Loaders.ipynb
│   ├── reports/
│   │   ├── DATA_EXPLORATION_REPORT.md
│   │   ├── DATA_PREPROCESSING_REPORT.md
│   │   └── DATA_LOADERS_REPORT.md
│   └── src/
│       └── data_loader.py             # Reusable TF data loader package
│
├── 🤖 model_training/                 # Labani Nabila Nour El Houda — Phases 4–7
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
│       ├── densenet121_best_model.keras   (34.3 MB) ⭐
│       ├── resnet50_best_model.keras      (102.6 MB)
│       └── vgg16_best_model.keras         (57.7 MB)
│
├── 🖥️ app/                            # Developer & Business Model — Miloudi Maroua Amira
│   ├── app.py                         # Entry point, session gate, navigation hub
│   ├── profile_db.py                  # Auth boundary & session management
│   ├── patient_db.py                  # Patient CRUD: CSV + PDF + folder scaffolding
│   ├── vitaldiagnostique_db.py        # Vital sign diagnostic persistence
│   ├── xraydiagnostique_db.py         # X-ray image storage
│   ├── treatment_db.py                # Treatment CSV + PDF generation
│   ├── whyvitals.py                   # Medical explanation engine (Kassouar Fatima)
│   ├── whyxray.py                     # Radiology explanation dictionary
│   └── pages/
│       ├── Add_Patient.py             # Patient registration form
│       ├── vitaldiagnostique.py       # Vital signs form + GBM inference
│       ├── xraydiagnostique.py        # X-ray upload + DenseNet121 + GradCAM
│       ├── Treatment.py               # Treatment mode selector
│       ├── Hospitaltreatment.py       # Hospital admission + nurse monitoring
│       ├── profile.py                 # Doctor profile management
│       └── About_Us.py                # Team information
│
├── agentt/                            # Intelligent agent modules (NEW v9)
│   ├── layer1.py                      # Document ingestion: text extraction, chunking, FAISS + BM25
│   ├── retrieval.py                   # Hybrid cross-KB retrieval + Cohere/FlashRank reranking
│   ├── prompt_builder.py              # Structured clinical prompt with inference heuristics
│   ├── layer2.py                      # Chatbot orchestration + OpenRouter LLM streaming
│   ├── report_agent.py                # Full-patient PDF report generator (ReportLab)
│   ├── med_kb.py                      # Medical literature KB: build, cache, search
│   ├── agent_utils.py                 # Shared CSV/folder access utilities
│   ├── requirements.txt               # Agent-specific dependencies
│   ├── med_kb_cache.pkl               # Auto-generated KB cache (gitignore recommended)
│   └── refrences/                     # Medical literature source files
│       ├── diagnostics-15-02244.pdf
│       └── ijerph-17-06278.pdf
│
├── 🤖 models/
│   ├── densenet121_best_model.keras   # Fine-tuned CNN (97.21% external sensitivity)
│   └── Gradient_Boost.pkl            # GBM classifier (Kassouar Fatima)
│
├── 📊 data/                           # Datasets (not included — see Quick Start)
│   └── README.md
│
├── 📁 docs/
│   ├── images/                        # All visualizations
│   │   ├── 01-09_*.png               # Data engineering & training figures
│   │   └── xray-heart.gif            # Banner
│   ├── Phase2_Final (Repaired).pdf    # Model training & evaluation report
│   ├── Preprocessing_Report_Final.pdf # Data preprocessing report
│   └── TOPIC_PROPOSAL_2.pdf          # Original project proposal
│
├── ⚙️ config/
│   ├── preprocessing_config.json
│   └── data_loaders_summary.json
│
├── 📖 README.md
├── 📦 requirements.txt
└── 📄 LICENSE
```

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install tensorflow>=2.10 keras numpy matplotlib scikit-learn seaborn opencv-python \
            streamlit pillow reportlab joblib
```

### 1. Clone the Repository

```bash
git clone https://github.com/AminaMar/pediatric-pneumonia-detection.git
cd pediatric-pneumonia-detection
```

### 2. Download the Dataset

- [Kaggle Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Place in: `data/chest_xray/`

### 3. Run the Data Engineering Pipeline *(Optional — already completed)*

```bash
# Run notebooks in order:
data_engineering/notebooks/01_Data_Exploration_EDA.ipynb
data_engineering/notebooks/02_Data_Preprocessing.ipynb
data_engineering/notebooks/03_Data_Loaders.ipynb
```

### 4. Train Models *(Optional — pre-trained models available)*

```python
# Open in Google Colab (T4 GPU recommended)
model_training/notebooks/01_DenseNet121_Training.ipynb
model_training/notebooks/04_External_Validation.ipynb
model_training/notebooks/05_GradCAM.ipynb
```

### 5. Load Pre-trained Model

```python
import tensorflow as tf

# Download from Google Drive (link below)
model = tf.keras.models.load_model('models/densenet121_best_model.keras')

# Predict on a new X-ray
prediction = model.predict(preprocessed_image)
# Output > 0.260 → PNEUMONIA; output ≤ 0.260 → NORMAL
```

### 6. Launch the Clinical Application

```bash
streamlit run app/app.py
```

The app opens at `http://localhost:8501`. On first launch, complete your **Doctor Profile** to unlock all clinical features.

---

## 📖 Documentation

| Document | Description |
|---|---|
| [Data Preprocessing Report](docs/Preprocessing_Report_Final.pdf) | Split strategy, class weights, augmentation pipeline |
| [Model Training & Evaluation Report](docs/Phase2_Final%20(Repaired).pdf) | Model architecture, training details, threshold optimization, GradCAM |
| [Topic Proposal](docs/TOPIC_PROPOSAL_2.pdf) | Original project proposal |

---

## 💾 Pre-trained Models

Model files exceed GitHub's 25MB limit and are hosted on **Google Drive**.

👉 [**Download All Models (Google Drive)**](https://drive.google.com/drive/folders/1JtnqNL4lMSRHBtR_eex96k64wSix97Y9?usp=sharing)

| Model | Accuracy | Sensitivity | AUC-ROC | Size | Use |
|---|---|---|---|---|---|
| **densenet121_best_model.keras** | **94.31%** | **95.01%** | **0.981** | 34.3 MB | ⭐ **Recommended** |
| vgg16_best_model.keras | 91.35% | 95.32% | 0.964 | 57.7 MB | ⚠️ Acceptable |
| resnet50_best_model.keras | 78.16% | 95.48% | 0.880 | 102.6 MB | ❌ Not recommended |

---

## 🎓 Academic Context

- **Course:** AI in Healthcare
- **Institution:** University of Saida, Algeria
- **Academic Year:** 2025–2026
- **Supervisor:** Dr. Abderrahmane Khiat
- **Medical Advisor:** Dr. Aimer Mohammed Djamel Eddine (CHU Saida)

### Team Contribution Summary

| Member | Role | Key Deliverables |
|---|---|---|
| **Bouhmidi Amina Meroua** | Data Engineer | EDA, 70/15/15 stratified split, scientifically computed class weights (1.850/0.685), anatomically-constrained augmentation, optimized TF data loaders |
| **Labani Nabila Nour El Houda** | Deep Learning Engineer | 3 trained CNN models (VGG16/ResNet50/DenseNet121), ROC threshold optimization (0.260), GradCAM clinical heatmaps, model comparison, external validation (**97.21% sensitivity**) |
| **Miloudi Maroua Amira** |Generative AI Engineer & Developer & Business Model| Complete RAG & PDF outomation , Streamlit clinical application, UI/UX design, EMR integration strategy, business model & deployment |
| **Kassouar Fatima** | Manager | Gradient Boosting vital signs classifier, CSV preprocessing pipeline and management |
| Dr. Abderrahmane Khiat | Academic Supervisor | Academic supervision & evaluation |

---

## 📚 References

1. **CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning** — *Stanford AI Lab (Rajpurkar et al., 2017)*
2. **Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization** — *Selvaraju et al., 2017*
3. **Detection of Pneumonia in Children Through Chest Radiographs Using AI in Low-Resource Settings** — *PLOS Digital Health, 2025*
4. **AI–EHR Integration Improving Diagnostic Capabilities Through HL7/FHIR Standards** — *PMC/PubMed Central, 2024*
5. **WHO IMCI: Integrated Management of Childhood Illness Guidelines** — *World Health Organization*
6. **IDSA/ATS Consensus Guidelines on Community-Acquired Pneumonia** — *Infectious Diseases Society of America*
7. **BTS Guidelines for the Management of Community Acquired Pneumonia in Children** — *British Thoracic Society*
8. **Pneumonia in Children — Fact Sheet** — *WHO, 2024*

---

## 📊 Key Achievements Summary

✅ **Clinical Targets Met (DenseNet121):**
- Accuracy: 94.31% (target: ≥90%)
- Sensitivity: 95.01% (target: ≥95%) — improved to **97.21%** on external data
- Specificity: 92.44% (target: ≥85%)

✅ **Data Engineering Impact:**
- Validation set increased **54×** (16 → 878 images)
- Scientifically computed class weights via sklearn
- Anatomically-constrained augmentation pipeline

✅ **Complete End-to-End Clinical System:**
- Full pipeline: raw data → preprocessing → model training → explainability → deployment
- Dual AI diagnostic system: X-ray (Deep Learning) + Vital Signs (Machine Learning)
- Fully functional Streamlit-based clinical application used as a real decision support tool
- Zero-infrastructure architecture — deployable in any clinic without a database

✅ **RAG & Intelligent Automation Integration (Major Contribution):**
- Implementation of a Retrieval-Augmented Generation (RAG) clinical chatbot
- Combines patient data + medical literature for evidence-based reasoning
- Hybrid retrieval: FAISS + BM25 + reranking
- Automated clinical PDF report generation including patient identity, medical case, diagnostics, and treatment plans
- Full patient data lifecycle management
- Structured storage of history, diagnostics, treatments, and monitoring records in files
- Intelligent automation of clinical reasoning assistance, report generation, and data organization and traceability

✅ **Enhanced Clinical User Experience (UX Innovation):**
- Emergency protocol system (Doctor → Emergency Doctor handover workflow)
- Parental guidance module with clear instructions for handling critical and high-risk situations at home
- Real-time nurse monitoring system with structured tracking tables
- Explainable AI outputs with clinical justification, not just predictions

✅ **Documentation & Professional Delivery:**
- 3 detailed data engineering reports
- Complete notebooks for all ML/DL phases
- Full academic final report
- Structured project presentations and technical documentation

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

*Developed as an academic capstone at the University of Saida, Algeria, Academic Year 2025–2026.*

---

## 📧 Contact

**Data Engineer:** Bouhmidi Amina Meroua — [GitHub @AminaMar](https://github.com/AminaMar)  
**Deep Learning Engineer:** Labani Nabila Nour El Houda — [GitHub @labaninabila193-code](https://github.com/labaninabila193-code)  
**Application Developer & Business Model:** Miloudi Maroua Amira — [GitHub @lily01011](https://github.com/lily01011)


**Project Repository:** [github.com/AminaMar/pediatric-pneumonia-detection](https://github.com/AminaMar/pediatric-pneumonia-detection)

---

<div align="center">

**University of Saida, Algeria — 2025–2026**

*This project addresses a critical healthcare challenge in Algeria through state-of-the-art AI,*
*aligned with the National Digital Health Strategy.*

*Complete pipeline: Data Engineering · Deep Learning · Clinical Application · Project Management*

</div>
readme.md
Displaying readme.md.
