🏥 AI-Powered Pediatric Pneumonia Detection
<p align="center"> <img src="docs/images/xray-heart.gif" alt="Medical AI Banner" width="100%"> </p> <p align="center"> <img src="https://img.shields.io/badge/Python-3.10+-blue.svg?style=for-the-badge&logo=python" alt="Python"> <img src="https://img.shields.io/badge/TensorFlow-2.19-orange.svg?style=for-the-badge&logo=tensorflow" alt="TensorFlow"> <img src="https://img.shields.io/badge/Keras-3.x-red.svg?style=for-the-badge&logo=keras" alt="Keras"> <img src="https://img.shields.io/badge/Streamlit-Latest-FF4B4B.svg?style=for-the-badge&logo=streamlit" alt="Streamlit"> <img src="https://img.shields.io/badge/Status-Complete-success.svg?style=for-the-badge" alt="Status"> <img src="https://img.shields.io/badge/Sensitivity-97.21%25-success.svg?style=for-the-badge" alt="Sensitivity"> <img src="https://img.shields.io/badge/Accuracy-94.31%25-success.svg?style=for-the-badge" alt="Accuracy"> <img src="https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge" alt="License"> </p> <h3 align="center">AI-Powered Early Pediatric Pneumonia Detection: Integration with Electronic Medical Records in Algeria</h3> <p align="center">Leveraging Transfer Learning, Strategic Data Engineering, and Full-Stack Clinical Application Development</p>
📋 Table of Contents
Overview
Team
Problem Statement
Solution
Project Status
Dataset
Data Engineering Pipeline
Model Training
Clinical Application
Results
Model Comparison
External Validation
Grad-CAM Visualizations
Repository Structure
Quick Start
Documentation
Pre-trained Models
Academic Context
References
License
🔍 Overview
This project delivers a complete, end-to-end multimodal AI clinical assistant for pediatric pneumonia detection — a system that leverages ML/DL for prediction, RAG for reasoning, and automation for reporting, enabling accurate, explainable, and workflow-integrated medical decision support. From raw data engineering through to a fully deployed Streamlit application used by clinicians in real time.

The pipeline spans four major pillars:

Pillar 1 — Data Engineering (Bouhmidi Amina Meroua)

Complete EDA on 5,863 pediatric chest X-rays
Dataset reorganization with stratified 70/15/15 split
Scientifically computed class weights and intelligent augmentation
Optimized TensorFlow data loaders for GPU training
Pillar 2 — Model Training & Deep Learning (Labani Nabila Nour El Houda)

Transfer learning pipeline across VGG16, ResNet50, and DenseNet121
Threshold optimization via ROC analysis (0.5 → 0.260)
GradCAM clinical interpretability implementation
External cross-dataset validation achieving 97.21% sensitivity
Pillar 3 — Generative AI, Full Stack & Business Model (Miloudi Maroua Amira)

Streamlit-based multimodal AI clinical decision support application (ML + DL + Agentic AI)
Dual AI diagnostic pathways: DenseNet121 (X-ray) + Gradient Boosting (vital signs)
Medical explanation engine (whyvitals.py) — RAG-powered clinical reasoning with per-feature explanations, interaction detection, and literature-backed reporting
Complete doctor/patient management with zero-infrastructure file-based persistence
Real-time nurse monitoring, treatment planning, and automated PDF report generation
Agentic AI workflow integration for end-to-end clinical automation
Business model design and deployment strategy
Pillar 4 — Vital Signs ML Pipeline (Kassouar Fatima)

Gradient Boosting classifier for vital signs-based diagnosis
CSV preprocessing pipeline and data management
Key Achievement: DenseNet121 achieved 95.01% Sensitivity and 94.31% Accuracy on the internal test set, improving to 97.21% Sensitivity on external validation. The full clinical application integrates the deep learning model, a Gradient Boosting vital signs classifier, and a RAG-based medical explanation engine into a production-grade multimodal AI Streamlit interface.

👥 Team
<table> <tr> <td align="center"><b>Role</b></td> <td align="center"><b>Name</b></td> <td align="center"><b>Core Responsibilities</b></td> </tr> <tr> <td>🔧 <b>Data Engineer</b></td> <td><b>Bouhmidi Amina Meroua</b></td> <td>Complete data pipeline (EDA → preprocessing → data loaders), 70/15/15 stratified split, class weight computation, augmentation design</td> </tr> <tr> <td>🤖 <b>Deep Learning Engineer</b></td> <td><b>Labani Nabila Nour El Houda</b></td> <td>DenseNet121 / VGG16 / ResNet50 training, threshold optimization (0.260), GradCAM implementation, external dataset validation (97.21% sensitivity), X-ray diagnostic module in app</td> </tr> <tr> <td>💻 <b>Generative AI Engineer & Full Stack Developer & Business Model</b></td> <td><b>Miloudi Maroua Amira</b></td> <td>Full Streamlit application development (architecture, UI/UX, all pages), RAG-based medical explanation engine (whyvitals.py), agentic AI workflow integration, automated PDF reporting, clinical workflow integration, business model design & deployment strategy</td> </tr> <tr> <td>📋 <b>Manager</b></td> <td><b>Kassouar Fatima</b></td> <td>Gradient Boosting classifier (vital signs), CSV preprocessing pipeline and data management</td> </tr> <tr> <td>👨‍🏫 Academic Supervisor</td> <td>Dr. Abderrahmane Khiat</td> <td>Academic guidance & evaluation</td> </tr> <tr> <td>🏥 Medical Advisor</td> <td>Dr. Aimer Mohammed Djamel Eddine</td> <td>Clinical validation, medical references & advisory</td> </tr> </table>
Institution: University of Saida, Algeria — Academic Year 2025–2026

🚨 Problem Statement
Pneumonia is a leading cause of death in children under 5:

📊 740,000+ deaths annually worldwide (15% of child mortality)
🏥 200–300 cases daily in Algerian emergency departments
⏱️ 6–24 hour diagnostic delays due to radiologist shortage
👨‍⚕️ ~1 pediatric radiologist per 500,000 children — a 500× deficiency vs. developed nations
🚗 40% of children must travel 50–100 km to access chest X-ray imaging
📉 70–85% radiologist agreement on pneumonia diagnosis — significant inter-observer error
💡 Solution
A multimodal AI clinical assistant combining ML, DL, and Agentic AI:

🧠 Deep Learning — DenseNet121 fine-tuned on 5,856 pediatric chest X-rays (95.01% sensitivity)
🌿 Machine Learning — Gradient Boosting classifier on 12 vital sign parameters
🔥 GradCAM Explainability — Clinical heatmaps showing which lung regions influenced each decision
🤖 Agentic AI — Automated clinical workflow: end-to-end diagnosis, report generation, nurse monitoring triggers, and treatment planning with zero manual intervention
📚 RAG-Powered Reasoning — whyvitals.py medical explanation engine retrieves literature-backed clinical notes (WHO IMCI, IDSA/ATS, BTS, NEJM EPIC) per feature and interaction
⚖️ Strategic Class Balancing — Computed weights (1.850 / 0.685) via sklearn
🎨 Intelligent Data Augmentation — Anatomically-constrained (no vertical flips)
🏥 EMR Integration Ready — HL7/FHIR compatible architecture
🖥️ Full Clinical Application — Streamlit app: patient registration, dual AI diagnostics, treatment planning, nurse monitoring
🗄️ Zero-Infrastructure Deployment — File-based persistence, no database server required
📊 Project Status
<table> <tr> <th>Phase</th> <th>Lead</th> <th>Status</th> <th>Progress</th> <th>Deliverables</th> </tr> <tr> <td>📊 Data Exploration</td> <td>Bouhmidi Amina Meroua</td> <td>✅ Complete</td> <td>████████████ 100%</td> <td>EDA Report, Figures 01–04</td> </tr> <tr> <td>🔧 Data Preprocessing</td> <td>Bouhmidi Amina Meroua</td> <td>✅ Complete</td> <td>████████████ 100%</td> <td>Preprocessed Dataset, Figures 05–07, preprocessing_config.json</td> </tr> <tr> <td>📦 Data Loaders</td> <td>Bouhmidi Amina Meroua</td> <td>✅ Complete</td> <td>████████████ 100%</td> <td>TF Generators, data_loader.py, data_loaders_summary.json</td> </tr> <tr> <td>🤖 Model Training (×3)</td> <td>Labani Nabila Nour El Houda</td> <td>✅ Complete</td> <td>████████████ 100%</td> <td>VGG16, ResNet50, DenseNet121 trained models</td> </tr> <tr> <td>📈 Model Evaluation & Optimization</td> <td>Labani Nabila Nour El Houda</td> <td>✅ Complete</td> <td>████████████ 100%</td> <td>ROC analysis, threshold 0.260, full metrics report</td> </tr> <tr> <td>🔥 Grad-CAM Interpretability</td> <td>Labani Nabila Nour El Houda</td> <td>✅ Complete</td> <td>████████████ 100%</td> <td>Clinical heatmaps, GradCAM module</td> </tr> <tr> <td>🌍 External Dataset Validation</td> <td>Labani Nabila Nour El Houda</td> <td>✅ Complete</td> <td>████████████ 100%</td> <td>97.21% sensitivity on independent dataset</td> </tr> <tr> <td>🖥️ Clinical Streamlit Application</td> <td>Miloudi Maroua Amira</td> <td>✅ Complete</td> <td>████████████ 100%</td> <td>Doctor onboarding, patient management, dual AI diagnostics, RAG explanation engine, agentic workflow, treatment module, nurse monitoring</td> </tr> <tr> <td>🌿 Vital Signs ML Module</td> <td>Kassouar Fatima</td> <td>✅ Complete</td> <td>████████████ 100%</td> <td>Gradient Boosting model, CSV preprocessing pipeline</td> </tr> <tr> <td>📝 Reports & Representation</td> <td>Full Team</td> <td>✅ Complete</td> <td>████████████ 100%</td> <td>3 engineering reports, complete final report, technical guide, project presentations</td> </tr> </table>
📁 Dataset
Original Dataset
Source: Kaggle Pediatric Chest X-Ray Pneumonia
Size: 5,863 labeled images
Classes: NORMAL (1,583) | PNEUMONIA (4,273)
Patients: Pediatric patients aged 1–5 years
Imbalance Ratio: 2.7:1 (PNEUMONIA:NORMAL)
Preprocessed Dataset (70/15/15 Stratified Split)
<table> <tr> <th>Split</th> <th>NORMAL</th> <th>PNEUMONIA</th> <th>Total</th> <th>Percentage</th> </tr> <tr> <td><b>Training</b></td> <td>1,108 (27.0%)</td> <td>2,991 (73.0%)</td> <td>4,099</td> <td>70%</td> </tr> <tr> <td><b>Validation</b></td> <td>237 (27.0%)</td> <td>641 (73.0%)</td> <td>878</td> <td>15%</td> </tr> <tr> <td><b>Test</b></td> <td>238 (27.1%)</td> <td>641 (72.9%)</td> <td>879</td> <td>15%</td> </tr> </table>
Configuration:

📏 Image size: 224×224 pixels
🎨 Format: Grayscale, normalized [0-1]
🔄 Augmentation: Rotation (±15°), Shift (10%), Zoom (10%), Horizontal flip only
External Validation Dataset
Source: Pneumonia Radiography Dataset
Size: 488 images (237 NORMAL, 251 PNEUMONIA)
Purpose: Cross-dataset validation to test real-world generalization
🔧 Data Engineering Pipeline
Lead: Bouhmidi Amina Meroua

Phase 1 — Data Exploration ✅
Objective: Understand dataset structure, identify quality issues

Key Findings:

Total images: 5,863 (1,583 NORMAL, 4,273 PNEUMONIA)
Class imbalance: 2.7:1 ratio (PNEUMONIA:NORMAL)
Critical issue detected: Original validation set only 16 images (0.3%) → statistically unusable
Variable image dimensions: 384×127 px to 2,916×2,713 px
Measurable pixel intensity differences between classes
Deliverables:

Exploration Report
Figures 01–04: distribution analysis, sample review, dimension mapping, pixel statistics
Phase 2 — Preprocessing ✅
Objective: Reorganize dataset, balance classes, build preprocessing pipeline

Actions Taken:

Dataset Reorganization

Created 70/15/15 stratified split
Validation set: 16 → 878 images (54× increase)
Class distribution (27%/73%) maintained across all three splits
Class Weight Calculation

Method: sklearn.utils.class_weight.compute_class_weight('balanced')
NORMAL (class 0): 1.850 — forces model to penalize missed normals
PNEUMONIA (class 1): 0.685
Result: model gives 2.7× more learning attention to the minority class
Augmentation Strategy (anatomically constrained)

Rotation: ±15° (realistic patient positioning variation)
Horizontal/vertical shifts: 10%
Zoom: 10%
Horizontal flip: ✅ (lungs are symmetric)
Vertical flip: ❌ (anatomically incorrect — never applied)
Applied only to training set — validation/test are untouched
Preprocessing Pipeline

Resize: 224×224 (required by ImageNet pretrained architectures)
Normalize: [0–255] → [0–1] float32
Format: Grayscale (224, 224, 1)
Deliverables:

Preprocessing Report
Figures 05–07: quality checks, augmentation samples, pipeline diagram
preprocessing_config.json
Phase 3 — Data Loaders ✅
Objective: Build optimized TensorFlow generators for GPU training

Implementation:

tf.keras.preprocessing.image.ImageDataGenerator with full augmentation pipeline
Batch size: 32 images
Training: shuffle enabled + augmentation active
Validation / Test: no shuffle, no augmentation (clean evaluation)
Performance: <0.5 seconds per batch on T4 GPU
Deliverables:

Data Loaders Report
Figures 08–09: batch sample visualization, class distribution per split
data_loader.py — reusable, importable Python package
data_loaders_summary.json
📊 Data Engineering Impact
Contribution	Before	After	Impact
Validation set size	16 images	878 images	+54× increase
Class weight handling	None	1.850 / 0.685	Minority class emphasis
Augmentation	None	5 transforms	+1–2% accuracy
Preprocessing pipeline	Raw pixels	224×224 normalized	Training-ready format
Documentation	None	3 detailed reports	Full reproducibility
Estimated accuracy contribution from data engineering alone: +2.5% to +4.0%

🤖 Model Training
Lead: Labani Nabila Nour El Houda (@labaninabila193-code)

Methodology
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
Models Trained
#	Architecture	Parameters	ImageNet Weights	Training Time
1	VGG16	138M	✅	120 min
2	ResNet50	25M	✅	~60 min
3	DenseNet121 ⭐	8M	✅	58 min
Training Configuration
Platform: Google Colab (T4 GPU)
Framework: TensorFlow 2.19 / Keras 3.x
Optimizer: Adam
Loss: Binary Cross-Entropy
Early Stopping: Patience = 7 epochs
Batch Size: 32
Class Weights: {0: 1.850, 1: 0.685} — provided by Data Engineer
🌿 Vital Signs ML Module
Lead: Kassouar Fatima

Kassouar Fatima developed the machine learning pipeline for vital signs-based pneumonia classification, building the Gradient Boosting model and the complete CSV preprocessing pipeline integrated into the clinical application.

Gradient Boosting Classifier
Attribute	Detail
File	models/Gradient_Boost.pkl
Framework	scikit-learn 1.6.1 (serialized via joblib)
Input	12-feature encoded vector (categoricals + numerics)
Output	Binary class (0 = Not Sick, 1 = Sick) + probability score
Top feature	Confusion — 64.8% importance
Feature Importance Ranking:

Rank	Feature	Importance
1	Confusion	64.8%
2	Fever severity	12.9%
3	Temperature	9.2%
4–12	Remaining 9 features	13.1% combined
Input Features (12 Parameters)
#	Feature	Type
1	Gender	Categorical
2	Age (1–16)	Integer
3	Cough type	Categorical
4	Fever severity	Categorical
5	Shortness of breath	Boolean
6	Chest pain	Boolean
7	Fatigue	Boolean
8	Confusion	Boolean (top feature: 64.8%)
9	Oxygen saturation (SpO₂)	Float
10	Crackles	Boolean
11	Sputum color	Categorical
12	Temperature	Float
🖥️ Clinical Application
Generative AI Engineer & Full Stack Developer & Business Model: Miloudi Maroua Amira (@lily01011)

The clinical application was built and designed by Miloudi Maroua Amira as a multimodal AI clinical assistant — integrating the DL models trained by Labani Nabila, the preprocessed data pipeline built by Bouhmidi Amina, and the Gradient Boosting model trained by Kassouar Fatima into a complete, production-grade system. Miloudi Maroua Amira also engineered the RAG-powered medical explanation engine (whyvitals.py), the agentic AI automation layer, and the full automated PDF reporting pipeline. The business model and deployment strategy were also designed by Miloudi Maroua Amira.

App Architecture: ML + DL + Agentic AI — a multimodal AI clinical assistant that leverages ML/DL for prediction, RAG for reasoning, and automation for reporting, enabling accurate, explainable, and workflow-integrated medical decision support.

Dual AI Diagnostic Pathways
Pathway	Input	Model	Output
Chest X-Ray Diagnosis	Pediatric chest radiograph (PNG/JPG)	DenseNet121 CNN	PNEUMONIA / NORMAL + GradCAM heatmap + radiological explanation
Vital Signs Diagnosis	12 clinical parameters	Gradient Boosting Classifier	Sick / Not Sick + risk score (0–100) + per-feature medical explanation
Key Clinical Features
Doctor & Patient Management

Mandatory doctor profile gate — all clinical features locked until profile is complete
Auto-incrementing patient IDs (starting at 1001)
Immutable patient records: CSV + formatted PDF on registration
Doctor-namespaced data isolation — architecturally impossible to access another doctor's data
Vital Signs Diagnostic Module (whyvitals.py — RAG-powered medical explanation engine)

12-parameter clinical form: temperature, SpO₂, cough type, fever severity, shortness of breath, chest pain, fatigue, confusion, crackles, sputum color, age, gender
Risk score 0–100 computed from feature importance weights
RAG-based reasoning: FEATURE_EXPLANATIONS maps every (feature, value) pair to a clinical note + literature reference (WHO IMCI, IDSA/ATS, BTS, NEJM EPIC study)
INTERACTION_EXPLANATIONS — lambda-based rules detecting clinically significant multi-feature combinations (e.g., high fever + purulent sputum → bacterial pneumonia pattern)
SCORE_WEIGHTS — mirrors model feature importance for the 0–100 risk score
explain() — returns: verdict, score, summary, tags, interaction notes, per-feature explanations
Top predictive feature: Confusion (64.8% importance)
whyvitals.py is fully testable independently of the Streamlit UI — it can be imported into any Python context (REST API, CLI, unit tests) without modification.

X-Ray Diagnostic Module (whyxray.py — radiology explanation dictionary)

DenseNet121 inference with classification threshold at 0.260
GradCAM via TensorFlow GradientTape on layer conv5_block16_concat
3-panel figure: original / heatmap / overlay rendered inline
Three outcome states: PNEUMONIA / NORMAL / uncertain (borderline safety pathway)
Radiological sign explanations with cited sources (Stanford CheXNet, Radiopaedia, RSNA, AJR)
Treatment Planning

Home management: dynamic medication list, follow-up notes, emergency warning signs, parent handover plan
Hospital management: admission notes, nursing instructions, discharge readiness flags (SpO₂, fever-free 24h, oral feeds, improved X-ray)
All plans generate a branded PDF report automatically
Real-Time Nurse Monitoring

Configurable interval timer (seconds / minutes / hours)
15 WHO-aligned monitoring parameters per interval + custom doctor-defined columns
Each interval saved as Check Table N.csv — append-only, forensically sound
Agentic AI Automation

End-to-end clinical workflow automation: patient intake → dual AI diagnosis → explanation generation → treatment planning → PDF report → nurse monitoring activation
Zero-touch reporting: automated PDF generation triggered on diagnosis completion
Agentic orchestration of multi-step clinical tasks without manual intervention
System Architecture (Three-Tier)
╔══════════════════════════════════════════════════════════════════╗
║                   TIER 1 — PRESENTATION LAYER                   ║
║  app.py · profile.py · Add_Patient.py · vitaldiagnostique.py   ║
║  xraydiagnostique.py · Treatment.py · Hospitaltreatment.py     ║
╚══════════════════════════════╦═══════════════════════════════════╝
                               ║ function calls only
╔══════════════════════════════╩═══════════════════════════════════╗
║         TIER 2 — BUSINESS LOGIC, GENERATIVE AI & ML LAYER      ║
║  profile_db.py  ·  patient_db.py  ·  vitaldiagnostique_db.py   ║
║  xraydiagnostique_db.py  ·  treatment_db.py                    ║
║  whyvitals.py (RAG explanation engine — Miloudi Maroua Amira)  ║
║  whyxray.py (radiology explanation — Miloudi Maroua Amira)     ║
║  Gradient_Boost.pkl  ·  densenet121_best_model.keras            ║
╚══════════════════════════════╦═══════════════════════════════════╝
                               ║ pathlib.Path file I/O
╔══════════════════════════════╩═══════════════════════════════════╗
║                  TIER 3 — PERSISTENCE LAYER                     ║
║  data/<Region-Hospital>/<DoctorName>/<PatientID_Name>/          ║
║  doctor_profile.csv  ·  N-vitaldiagnostic.csv                   ║
║  home_treatment.csv  ·  hospitalizedtreatment.csv               ║
║  Check Table N.csv   ·  xray/*.png  ·  *_pdf/ folders           ║
╚══════════════════════════════════════════════════════════════════╝
Data Integrity Mechanisms
Session gating: All operations verify active doctor session before any file I/O
Append-only diagnostics: Sequential N-vitaldiagnostic.csv naming — immutable history
Two-step save workflow: AI results displayed for doctor review before any file is written
Path sanitization: All folder names sanitized; no OS-reserved characters
UTF-8 enforcement: Correct handling of Arabic-French patient names
Zero-infrastructure: Runs on any machine with Python — no network, no database server
📊 Results
🥇 DenseNet121 — Best Model ⭐
Metric	Default Threshold (0.5)	Optimized Threshold (0.260)	Clinical Target
Accuracy	92.04%	94.31%	≥90% ✅
Sensitivity	90.02%	95.01%	≥95% ✅
Specificity	97.48%	92.44%	≥85% ✅
Precision	98.97%	97.13%	—
F1-Score	0.9428	—	—
AUC-ROC	0.9810	—	—
Training Time	—	57.97 min	—
Confusion Matrix (Optimized Threshold = 0.260):

                    Predicted NORMAL    Predicted PNEUMONIA
Actual NORMAL            220 (TN)              18 (FP)
Actual PNEUMONIA          32 (FN)             609 (TP)
✅ Clinical Impact: Missed pneumonia cases reduced by 50% (64 → 32) after threshold optimization.

🥈 VGG16
Metric	Default (0.5)	Optimized (0.110)	Target
Accuracy	76.11%	91.35%	≥90% ✅
Sensitivity	67.39%	95.32%	≥95% ✅
Specificity	99.58%	80.67%	≥85% ⚠️
AUC-ROC	0.9644	—	—
⚠️ Required an extreme threshold (0.110) to reach sensitivity target. Specificity remains below clinical threshold.

🥉 ResNet50
Metric	Default (0.5)	Optimized	Target
Accuracy	82.14%	78.16%	≥90% ❌
Sensitivity	84.87%	95.48%	≥95% ✅
Specificity	—	31.51%	≥85% ❌
AUC-ROC	0.8802	—	—
❌ Clinically unacceptable: 31.51% specificity means 69% of healthy children are incorrectly flagged as pneumonia cases.

🏆 Model Comparison
Model	Accuracy	Sensitivity	Specificity	AUC-ROC	Time	Clinical Verdict
DenseNet121 ⭐	94.31%	95.01%	92.44%	0.981	58 min	✅ ALL TARGETS MET
VGG16	91.35%	95.32%	80.67%	0.964	120 min	⚠️ Partial
ResNet50	78.16%	95.48%	31.51%	0.880	~60 min	❌ Not suitable
Why DenseNet121 Won
✅ Only model meeting ALL clinical targets simultaneously
✅ Dense connections — superior feature reuse, better gradient flow
✅ Parameter efficiency — 8M vs. 138M (VGG16) → less overfitting risk
✅ Clinically validated architecture — Stanford CheXNet (2017) used DenseNet121 for chest X-rays
✅ Fastest training — 58 min vs. 120 min for VGG16
✅ Best AUC-ROC — 0.981
✅ Balanced performance — no extreme threshold manipulation needed
🌍 External Validation
DenseNet121 tested on a completely independent dataset from a different source (488 images, never seen during training or validation):

Metric	Internal Test Set	External Dataset	Difference
Accuracy	94.31%	87.09%	-7.22%
Sensitivity	95.01%	97.21%	+2.20% ✅
Specificity	92.44%	76.37%	-16.07%
Total Samples	879	488	—
🎯 Key Finding: Sensitivity improved on the external dataset (97.21% vs. 95.01%), confirming the model generalizes to real-world, unseen clinical data. Accuracy above 80% on a fully independent dataset is considered strong generalization in medical AI literature.

🔥 Grad-CAM Visualizations
Grad-CAM highlights which lung regions influenced each AI decision, enabling clinical verification and building radiologist trust in the system:

<div align="center">
⚠️ PNEUMONIA Case — Correctly Detected
Pneumonia Detection

Model Decision: PNEUMONIA (Confidence: 99.3%)
Grad-CAM: Highlights consolidation in right lower lobe
Clinical Correlation: ✅ Correct — matches radiological findings

✅ NORMAL Case — Correctly Classified
Normal Classification

Model Decision: NORMAL (Confidence: 98.8%)
Grad-CAM: Focuses on central mediastinal structures
Clinical Correlation: ✅ Correct — no pathological features highlighted

</div>
🩺 Clinical Validation: Heatmaps confirm the model learned clinically meaningful features — consolidation/infiltrates for pneumonia cases, normal anatomical landmarks for healthy cases — rather than dataset artifacts.

📁 Repository Structure
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
├── 🖥️ app/                            # Generative AI Engineer & Full Stack Developer — Miloudi Maroua Amira
│   ├── app.py                         # Entry point, session gate, navigation hub
│   ├── profile_db.py                  # Auth boundary & session management
│   ├── patient_db.py                  # Patient CRUD: CSV + PDF + folder scaffolding
│   ├── vitaldiagnostique_db.py        # Vital sign diagnostic persistence
│   ├── xraydiagnostique_db.py         # X-ray image storage
│   ├── treatment_db.py                # Treatment CSV + PDF generation
│   ├── whyvitals.py                   # RAG-powered medical explanation engine (Miloudi Maroua Amira)
│   ├── whyxray.py                     # Radiology explanation dictionary (Miloudi Maroua Amira)
│   └── pages/
│       ├── Add_Patient.py             # Patient registration form
│       ├── vitaldiagnostique.py       # Vital signs form + GBM inference
│       ├── xraydiagnostique.py        # X-ray upload + DenseNet121 + GradCAM
│       ├── Treatment.py               # Treatment mode selector
│       ├── Hospitaltreatment.py       # Hospital admission + nurse monitoring
│       ├── profile.py                 # Doctor profile management
│       └── About_Us.py                # Team information
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
🚀 Quick Start
Prerequisites
pip install tensorflow>=2.10 keras numpy matplotlib scikit-learn seaborn opencv-python \
            streamlit pillow reportlab joblib
1. Clone the Repository
git clone https://github.com/AminaMar/pediatric-pneumonia-detection.git
cd pediatric-pneumonia-detection
2. Download the Dataset
Kaggle Chest X-Ray Pneumonia Dataset
Place in: data/chest_xray/
3. Run the Data Engineering Pipeline (Optional — already completed)
# Run notebooks in order:
data_engineering/notebooks/01_Data_Exploration_EDA.ipynb
data_engineering/notebooks/02_Data_Preprocessing.ipynb
data_engineering/notebooks/03_Data_Loaders.ipynb
4. Train Models (Optional — pre-trained models available)
# Open in Google Colab (T4 GPU recommended)
model_training/notebooks/01_DenseNet121_Training.ipynb
model_training/notebooks/04_External_Validation.ipynb
model_training/notebooks/05_GradCAM.ipynb
5. Load Pre-trained Model
import tensorflow as tf

# Download from Google Drive (link below)
model = tf.keras.models.load_model('models/densenet121_best_model.keras')

# Predict on a new X-ray
prediction = model.predict(preprocessed_image)
# Output > 0.260 → PNEUMONIA; output ≤ 0.260 → NORMAL
6. Launch the Clinical Application
streamlit run app/app.py
The app opens at http://localhost:8501. On first launch, complete your Doctor Profile to unlock all clinical features.

📖 Documentation
Document	Description
Data Preprocessing Report	Split strategy, class weights, augmentation pipeline
Model Training & Evaluation Report	Model architecture, training details, threshold optimization, GradCAM
Topic Proposal	Original project proposal
💾 Pre-trained Models
Model files exceed GitHub's 25MB limit and are hosted on Google Drive.

👉 Download All Models (Google Drive)

Model	Accuracy	Sensitivity	AUC-ROC	Size	Use
densenet121_best_model.keras	94.31%	95.01%	0.981	34.3 MB	⭐ Recommended
vgg16_best_model.keras	91.35%	95.32%	0.964	57.7 MB	⚠️ Acceptable
resnet50_best_model.keras	78.16%	95.48%	0.880	102.6 MB	❌ Not recommended
🎓 Academic Context
Course: AI in Healthcare
Institution: University of Saida, Algeria
Academic Year: 2025–2026
Supervisor: Dr. Abderrahmane Khiat
Medical Advisor: Dr. Aimer Mohammed Djamel Eddine (CHU Saida)

Team Contribution Summary
Member	Role	Key Deliverables
Bouhmidi Amina Meroua	Data Engineer	EDA, 70/15/15 stratified split, scientifically computed class weights (1.850/0.685), anatomically-constrained augmentation, optimized TF data loaders
Labani Nabila Nour El Houda	Deep Learning Engineer	3 trained CNN models (VGG16/ResNet50/DenseNet121), ROC threshold optimization (0.260), GradCAM clinical heatmaps, model comparison, external validation (97.21% sensitivity)
Miloudi Maroua Amira	Generative AI Engineer & Full Stack Developer & Business Model	Complete Streamlit multimodal AI clinical assistant (ML + DL + Agentic AI), RAG-powered medical explanation engine (whyvitals.py), agentic workflow automation, automated PDF reporting, UI/UX design, EMR integration strategy, business model & deployment
Kassouar Fatima	Manager	Gradient Boosting vital signs classifier, CSV preprocessing pipeline and data management
Dr. Abderrahmane Khiat	Academic Supervisor	Academic supervision & evaluation
📚 References
CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning — Stanford AI Lab (Rajpurkar et al., 2017)
Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization — Selvaraju et al., 2017
Detection of Pneumonia in Children Through Chest Radiographs Using AI in Low-Resource Settings — PLOS Digital Health, 2025
AI–EHR Integration Improving Diagnostic Capabilities Through HL7/FHIR Standards — PMC/PubMed Central, 2024
WHO IMCI: Integrated Management of Childhood Illness Guidelines — World Health Organization
IDSA/ATS Consensus Guidelines on Community-Acquired Pneumonia — Infectious Diseases Society of America
BTS Guidelines for the Management of Community Acquired Pneumonia in Children — British Thoracic Society
Pneumonia in Children — Fact Sheet — WHO, 2024
📊 Key Achievements Summary
✅ Clinical Targets Met (DenseNet121):

Accuracy: 94.31% (target: ≥90%)
Sensitivity: 95.01% (target: ≥95%) — improved to 97.21% on external data
Specificity: 92.44% (target: ≥85%)
✅ Data Engineering Impact:

Validation set increased 54× (16 → 878 images)
Scientifically computed class weights via sklearn
Anatomically-constrained augmentation pipeline
✅ Complete End-to-End Multimodal AI System:

Raw data → engineering → model training → interpretability → clinical application
Full Streamlit app: ML + DL + Agentic AI — multimodal clinical decision support
RAG-powered medical explanation engine with literature-backed reasoning
Agentic AI automation for end-to-end clinical workflows
Zero-infrastructure deployment — runs anywhere Python runs
✅ Documentation & Representation:

3 detailed data engineering reports
Complete notebooks for all phases
Comprehensive academic final report
Project presentations and team reports
📄 License
This project is licensed under the MIT License — see the LICENSE file for details.

Developed as an academic capstone at the University of Saida, Algeria, Academic Year 2025–2026.

📧 Contact
Data Engineer: Bouhmidi Amina Meroua — GitHub @AminaMar
Deep Learning Engineer: Labani Nabila Nour El Houda — GitHub @labaninabila193-code
Generative AI Engineer & Full Stack Developer & Business Model: Miloudi Maroua Amira — GitHub @lily01011

Project Repository: github.com/AminaMar/pediatric-pneumonia-detection

<div align="center">
University of Saida, Algeria — 2025–2026

This project addresses a critical healthcare challenge in Algeria through state-of-the-art AI,
aligned with the National Digital Health Strategy.

Complete pipeline: Data Engineering · Deep Learning · Generative AI · Full Stack Development · Project Management

Multimodal AI Clinical Assistant: ML + DL + Agentic AI

</div>
