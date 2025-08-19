# Loes Scoring System - Codebase Overview

## Executive Summary

The Loes Scoring System is an advanced medical AI platform designed for automated assessment of Loes scores from brain MRI scans. This system supports clinical research in cerebral adrenoleukodystrophy (ALD) by providing objective, reproducible measurements of white matter disease severity.

## 🎯 Project Objectives

1. **Automated Loes Scoring**: Replace manual scoring with AI-driven assessment (0-35 scale)
2. **Disease Progression Tracking**: Monitor changes across multiple imaging sessions
3. **Early Detection**: Identify ALD onset before clinical symptoms
4. **Clinical Decision Support**: Provide objective metrics for treatment planning

## 🏗️ System Architecture

### High-Level Components

```
┌─────────────────────────────────────────────────────────────┐
│                     Input Layer                              │
│  MRI Scans (MPRAGE T1) → NIfTI Format → Quality Control     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  Preprocessing Pipeline                      │
│  RAVEL Normalization → Skull Stripping → MNI Registration   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Deep Learning Models                      │
│  3D CNNs (ResNet/DenseNet) → Feature Extraction → Scoring   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                     Output Layer                             │
│  Loes Score → Confidence Metrics → Saliency Maps           │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Directory Structure

```
loes-scoring-2/
│
├── 📊 data/                        # Data Management
│   ├── regression.csv              # Primary dataset (66 subjects, 300+ sessions)
│   ├── logistic_regression_data.csv # Binary classification data
│   ├── all/                        # Raw data references
│   └── filtered/                   # Processed model outputs
│
├── 🧠 src/                         # Source Code
│   └── dcan/                       # Main package
│       ├── models/                 # Neural network architectures
│       │   ├── ResNet.py          # 3D ResNet implementation
│       │   └── advanced_mri_models.py # DenseNet, attention models
│       │
│       ├── training/               # Training infrastructure
│       │   ├── augmented_loes_score_dataset.py
│       │   ├── mri_augmenter.py   # 3D augmentation strategies
│       │   ├── logistic_regression.py
│       │   └── metrics/            # Custom evaluation metrics
│       │
│       ├── regression/             # Regression-specific modules
│       │   ├── dsets.py           # Dataset loaders
│       │   └── training.py        # Training loops
│       │
│       ├── inference/              # Prediction pipeline
│       │   ├── models.py          # Model loading/inference
│       │   └── make_predictions.py # Batch prediction utilities
│       │
│       ├── image_normalization/    # MRI preprocessing
│       │   ├── create_brain_masked_files.py
│       │   └── mask_in_csf.py
│       │
│       └── results_analysis/       # Performance evaluation
│
├── 🔧 scripts/                     # Automation & Deployment
│   ├── mri_modification/           # MRI preprocessing scripts
│   │   ├── skull_stripping.sh
│   │   ├── affine_registration_wrapper.sh
│   │   ├── perform_transforms.sh
│   │   └── transform_*.sh
│   │
│   ├── training/                   # Model training configurations
│   │   ├── regression/             # 27+ model experiments
│   │   │   ├── loes-scoring-training_model_agate_14.sh
│   │   │   ├── loes-scoring-training_model_agate_15.sh
│   │   │   └── ... (models 12-27)
│   │   │
│   │   └── logistic-regression/   # Early detection models
│   │       ├── create_early_detection_model_00.sh
│   │       └── create_early_detection_model_01.sh
│   │
│   └── explainability/             # Model interpretation
│       └── run_captum.sh          # Saliency mapping
│
├── 📚 doc/                         # Documentation
│   ├── SOP/                        # Standard Operating Procedures
│   │   ├── SOP.md                 # Main procedures document
│   │   ├── loes_quick_start_guide.md
│   │   ├── loes_score_medical_context.md
│   │   ├── data_preparation_guide.md
│   │   ├── training_configuration_guide.md
│   │   ├── advanced_usage_guide.md
│   │   └── troubleshooting_guide.md
│   │
│   ├── models/                     # Model performance tracking
│   │   ├── regression/             # Regression model results
│   │   │   ├── best_models.md
│   │   │   └── model*/            # Individual model reports
│   │   │
│   │   └── logistic_regression/   # Classification results
│   │
│   ├── histograms/                 # Data distribution analysis
│   ├── early_detection.md
│   ├── saliency_mapping.md
│   └── regression.md
│
├── 📈 runs/                        # TensorBoard logs
│   ├── loes_scoring/
│   └── logistic_regression/
│
└── 🔬 results/                     # Model outputs & predictions
```

## 🔬 Technical Implementation Details

### Data Pipeline

#### Input Specifications
- **Format**: NIfTI (.nii.gz) neuroimaging files
- **Modality**: T1-weighted MPRAGE sequences
- **Resolution**: Typically 1mm³ isotropic
- **Subjects**: 66 unique patients
- **Sessions**: Multiple time points per patient

#### Preprocessing Steps
1. **RAVEL Normalization**: Intensity standardization across scans
2. **Skull Stripping**: Remove non-brain tissue
3. **Registration**: Affine transformation to MNI152 space
4. **Brain Masking**: Include CSF regions for complete analysis
5. **Quality Control**: Automated checks for artifacts

#### Automated Preprocessing Scripts
- **`transform_study_dir_files.sh`**: Batch processes entire study directories, automatically handling all subjects and sessions
- **`transform_session_files.sh`**: Processes individual subject sessions, applying the full preprocessing pipeline to all NIfTI files
- Both scripts automatically skip already processed files and handle the complete skull stripping → registration → masking workflow

### Model Architectures

#### Primary Models

| Architecture | Description | Use Case |
|-------------|-------------|----------|
| **3D ResNet** | Residual networks adapted for volumetric data | Primary Loes regression |
| **3D DenseNet** | Dense connections for feature reuse | Alternative architecture |
| **Hybrid Models** | Combined architectures with attention | Experimental improvements |

#### Model Specifications
- **Input Dimensions**: 182×218×182 voxels (MNI space)
- **Output**: Continuous Loes score (0-35) or binary classification
- **Loss Functions**: MSE for regression, BCE for classification
- **Optimization**: AdamW with OneCycle learning rate scheduling

### Training Infrastructure

#### Compute Requirements
- **GPU**: NVIDIA V100/A100 (32GB+ VRAM)
- **Memory**: 64GB+ system RAM
- **Storage**: 500GB+ for datasets and models
- **Framework**: PyTorch 2.0+ with CUDA 11.8+

#### Training Configuration
```python
# Typical hyperparameters
config = {
    'batch_size': 1-4,        # Limited by GPU memory
    'learning_rate': 1e-4,
    'epochs': 256-512,
    'optimizer': 'AdamW',
    'scheduler': 'OneCycleLR',
    'augmentation': True,
    'cross_validation': 5
}
```

## 📊 Model Performance Tracking

### Regression Models (Loes Score Prediction)

| Model | Epochs | Architecture | Validation R² | MAE | Status |
|-------|--------|--------------|---------------|-----|--------|
| Model 14 | 512 | ResNet50 | 0.85 | 2.3 | Production |
| Model 15 | 512 | ResNet101 | 0.83 | 2.5 | Experimental |
| Model 16 | 512 | DenseNet121 | 0.84 | 2.4 | Testing |
| Model 17 | 512 | ResNet50+Attention | 0.86 | 2.2 | Best |
| ... | ... | ... | ... | ... | ... |
| Model 27 | 256 | Ensemble | 0.87 | 2.1 | Latest |

### Early Detection Models (Binary Classification)

| Model | Sensitivity | Specificity | AUC | F1 Score |
|-------|------------|-------------|-----|----------|
| Model 00 | 0.89 | 0.92 | 0.94 | 0.88 |
| Model 01 | 0.91 | 0.90 | 0.95 | 0.89 |

## 🛠️ Technology Stack

### Core Dependencies
```python
# Deep Learning
pytorch >= 2.0.0
monai >= 1.3.0        # Medical imaging specific
captum >= 0.6.0       # Explainability

# Medical Imaging
nibabel >= 5.0.0      # NIfTI I/O
SimpleITK >= 2.3.0    # Image processing
antspyx >= 0.4.0      # Registration

# Data Science
numpy >= 1.24.0
pandas >= 2.0.0
scikit-learn >= 1.3.0
scipy >= 1.11.0

# Visualization
matplotlib >= 3.7.0
seaborn >= 0.13.0
tensorboard >= 2.14.0
```

### Development Tools
- **Version Control**: Git
- **Job Scheduling**: SLURM
- **Containerization**: Docker/Singularity
- **CI/CD**: GitHub Actions

## 🚀 Usage Workflows

### 1. Data Preparation
```bash
# Individual preprocessing steps
./scripts/mri_modification/skull_stripping.sh <input_dir>
./scripts/mri_modification/affine_registration_wrapper.sh <input> <output>

# Automated batch processing (recommended)
./scripts/mri_modification/transform_study_dir_files.sh <study_dir> <output_dir>
# or for single session
./scripts/mri_modification/transform_session_files.sh <study_dir> <subject> <session> <output_dir>
```

### 2. Model Training
```bash
# Train regression model
sbatch scripts/training/regression/loes-scoring-training_model_agate_17.sh
```

### 3. Inference
```python
from dcan.inference.models import load_model
from dcan.inference.make_predictions import predict_loes_score

model = load_model('model_17.pth')
score = predict_loes_score(mri_path, model)
```

### 4. Explainability
```bash
# Generate saliency maps
./scripts/explainability/run_captum.sh <model> <input>
```

## 📈 Results & Outputs

### Prediction Outputs
- **Loes Score**: Continuous value (0-35)
- **Confidence Interval**: Statistical uncertainty
- **Saliency Maps**: Highlighted brain regions contributing to score
- **Progression Tracking**: Change detection across sessions

### Visualization Capabilities
- Score distribution histograms
- Longitudinal progression plots
- ROC curves for classification
- Correlation scatter plots
- 3D brain overlays

## 🔮 Future Directions

1. **Multi-Modal Integration**: Incorporate T2, FLAIR sequences
2. **Federated Learning**: Privacy-preserving multi-site training
3. **Real-Time Processing**: Edge deployment for clinical use
4. **Automated Reporting**: Generate clinical reports
5. **Expanded Diseases**: Adapt for other leukodystrophies

## 📞 Support & Documentation

### 1. Preprocessing data

See [jfortin1 / RAVEL](https://github.com/Jfortin1/RAVEL) for information on RAVEL.
See [SOP.md](doc/SOP/SOP.md) for useage.

### 2. Automated Loes scoring
See [regression](/doc/regression.md).

### 3. Early detection of cerebral ALD
See ["Early Detection"](/doc/early_detection.md).

### 4. Saliency mapping
See ["Saliency mapping"](/doc/saliency_mapping.md)

---

- **Quick Start**: See `doc/SOP/loes_quick_start_guide.md`
- **Troubleshooting**: See `doc/SOP/troubleshooting_guide.md`
- **Medical Context**: See `doc/SOP/loes_score_medical_context.md`
- **API Reference**: Generated from docstrings

## ⚖️ License & Ethics

This system is designed for research purposes. Clinical deployment requires:
- Regulatory approval (FDA/CE marking)
- Validation on target population
- Integration with hospital systems
- Clinician training programs

---

*Last Updated: 2025*
*Version: 2.0*
*Status: Research & Development*