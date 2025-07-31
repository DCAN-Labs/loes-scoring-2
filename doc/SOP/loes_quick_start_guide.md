# Quick Start Guide

## Prerequisites
- Python environment with PyTorch installed
- CUDA-capable GPU (recommended)
- At least 16GB RAM
- Sufficient storage for MRI data (typically 1-10GB per dataset)

## 1. Prepare Your Data

### Required Data Structure
Your project needs two main components:

#### CSV File Format
Create a CSV file with the following columns:
```csv
anonymized_subject_id,anonymized_session_id,scan,loes-score,Gd-enhanced
subject-00,session-00,mprage.nii.gz,1.0,0
subject-00,session-01,mprage.nii.gz,0.0,0
subject-01,session-00,mprage.nii.gz,6.5,0
```

**Column Descriptions:**
- `anonymized_subject_id`: Unique identifier for each patient
- `anonymized_session_id`: Unique identifier for each imaging session
- `scan`: Filename of the MRI scan (typically `.nii.gz` format)
- `loes-score`: Ground truth Loes score (0-34 scale)
- `Gd-enhanced`: Binary flag (0=no contrast, 1=contrast enhanced)

#### MRI Files Organization
Organize your NIFTI files in a single directory:
```
/path/to/your/data/
├── subject-00_session-00_space-MNI_brain_mprage_RAVEL.nii.gz
├── subject-00_session-01_space-MNI_brain_mprage_RAVEL.nii.gz
├── subject-01_session-00_space-MNI_brain_mprage_RAVEL.nii.gz
└── ...
```

**File Naming Convention:**
`{subject_id}_{session_id}_space-MNI_brain_mprage_RAVEL.nii.gz`

### Data Requirements
- **File Format**: NIFTI (.nii.gz)
- **Image Space**: MNI space (standardized brain template)
- **Preprocessing**: Images should be skull-stripped and intensity normalized
- **Sample Size**: Minimum 50-100 subjects recommended for training

### Critical: Prevent Data Leakage
**⚠️ Important**: Ensure no subject appears in both training and validation sets. All sessions from a given subject must be in the same split (training OR validation, never both).

## 2. Run Training with Minimal Configuration

### Basic Training Command
```bash
python src/dcan/training/regression.py \
    --csv-input-file data/your_data.csv \
    --folder /path/to/your/mri/files/ \
    --csv-output-file results/predictions.csv \
    --plot-location results/correlation_plot.png \
    --epochs 50 \
    "Quick start training"
```

### Expected Training Time
- **Small dataset** (50-100 subjects): 1-3 hours on GPU
- **Medium dataset** (200-500 subjects): 3-8 hours on GPU
- **Large dataset** (500+ subjects): 8+ hours on GPU

### What Happens During Training
1. **Data Loading**: Reads CSV and validates file paths
2. **Train/Validation Split**: Automatically splits subjects (80/20 by default)
3. **Model Training**: Trains ResNet model for specified epochs
4. **Validation**: Evaluates on held-out subjects
5. **Output Generation**: Creates model file, predictions CSV, and correlation plot

## 3. Evaluate Results

### Output Files
After training completes, you'll find:

#### Model File
- `model-{timestamp}.pt`: Trained PyTorch model
- Use for making predictions on new data

#### Predictions CSV
- Same format as input CSV plus `predicted_loes_score` column
- Only validation subjects will have predictions

#### Correlation Plot
- Scatter plot showing actual vs predicted Loes scores
- Includes correlation coefficient and fit line

#### Training Logs
Check the console output for:
- **Correlation coefficient**: Higher is better (>0.7 is good)
- **Standardized RMSE**: Lower is better
- **P-values**: Statistical significance of correlations

### Quick Quality Check
Good model performance indicators:
- Correlation coefficient > 0.7
- Points in scatter plot follow diagonal trend
- No obvious outliers or systematic bias

## Troubleshooting Common Issues

### File Not Found Errors
- Verify CSV file paths are correct
- Check that MRI file naming matches the pattern
- Ensure folder path ends with `/`

### CUDA/Memory Errors
- Reduce `--batch-size` (default: 32, try 16 or 8)
- Use `--DEBUG` flag to run on CPU for testing

### Poor Performance
- Increase `--epochs` (try 100-200)
- Ensure sufficient training data (>100 subjects)
- Check data quality and preprocessing

## Next Steps
Once basic training works:
1. **Optimize hyperparameters** (learning rate, batch size, model architecture)
2. **Implement cross-validation** for robust evaluation
3. **Experiment with data augmentation** for better generalization
4. **Deploy model** for clinical use