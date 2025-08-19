# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

This is a medical AI system for automated Loes scoring from brain MRI scans to assess cerebral adrenoleukodystrophy (ALD) severity. The system predicts scores on a 0-35 scale and performs early disease detection using 3D deep learning models.

## Essential Commands

### Environment Setup
```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Data Preprocessing Pipeline
```bash
# 1. Skull stripping (removes non-brain tissue)
./scripts/mri_modification/skull_stripping.sh <input.nii.gz> <output.nii.gz>

# 2. MNI registration (spatial normalization)
./scripts/mri_modification/affine_registration_wrapper.sh <input.nii.gz> <output.nii.gz>

# 3. Full preprocessing for a study directory
./scripts/mri_modification/transform_study_dir_files.sh <study_dir>

# 4. Brain masking (removes artifacts)
python src/dcan/image_normalization/create_brain_masked_files_from_csv.py --csv <input.csv>
```

### Model Training
```bash
# Train regression model (Loes score prediction)
python src/dcan/regression/training.py \
  --csv-input-file data/regression.csv \
  --batch-size 1 \
  --epochs 256 \
  --model resnet \
  --lr 0.0001 \
  --folder <mri_data_dir> \
  --model-save-location <output.pt>

# Submit SLURM job for HPC training
sbatch scripts/training/regression/loes-scoring-training_model_agate_17.sh
```

### Monitoring & Analysis
```bash
# Launch TensorBoard
tensorboard --logdir runs/

# Generate visualizations from predictions
python src/dcan/plot/create_scatterplot.py --csv <predictions.csv>
```

## Critical Architecture Patterns

### Data Flow Architecture
The system follows a strict pipeline: Raw MRI → Preprocessing → 3D CNN → Loes Score
- Input MRIs must be T1-weighted MPRAGE in NIfTI format (.nii.gz)
- All processing expects MNI152 space registration (182×218×182 voxels)
- Models consume full 3D volumes, not slices

### Model Architecture Hierarchy
```
src/dcan/models/
├── ResNet.py              # Primary 3D ResNet - PRODUCTION MODEL
├── advanced_mri_models.py  # DenseNet3D, experimental architectures
└── ../inference/models.py  # Model loading and prediction utilities
```

The ResNet architecture (Model 17) is the current best performer with R²=0.86.

### Dataset Management Pattern
All datasets inherit from `torch.utils.data.Dataset` with critical methods:
- `__getitem__`: Returns (image_tensor, loes_score, subject_id)
- Images are loaded via nibabel and converted to torch tensors
- Augmentation happens on-the-fly during training via `torchio`

### Training Configuration System
Training scripts in `scripts/training/regression/` follow naming: `loes-scoring-training_model_agate_XX.sh`
- Model numbers (12-27) represent evolutionary improvements
- Each script contains complete hyperparameter configuration
- SLURM headers specify GPU/memory requirements (typically 180GB RAM, 2xA100 GPUs)

### Cross-Validation Strategy
The system uses subject-level splits to prevent data leakage:
- Training/validation split at subject level, not scan level
- Multiple scans from same subject stay in same split
- Validation predictions saved to CSV for post-hoc analysis

## Key Implementation Details

### Memory Management
- Batch size limited to 1-4 due to 3D volume size
- Use gradient accumulation for effective larger batches
- Models checkpoint every N epochs to prevent loss from crashes

### File Naming Conventions
- Preprocessed files: `sub-XX_ses-YY_space-MNI_brain.nii.gz`
- Model checkpoints: `loes_scoring_XX.pt` where XX is model number
- Predictions: `modelXX.csv` with columns: subject_id, true_score, predicted_score

### GPU Utilization
- Always use DataParallel for multi-GPU training
- Mixed precision training (AMP) reduces memory by ~40%
- Gradient checkpointing trades compute for memory

### Error Handling Patterns
- All scripts check for existing outputs before processing
- Temporary files cleaned up on both success and failure
- SLURM error logs saved as `loes-scoring-*-%j.err`

## Common Pitfalls to Avoid

1. **Never mix Gd-enhanced and non-enhanced scans** - Filter using the 'Gd-enhanced' column
2. **Always preserve subject-level splits** - Scans from same patient must stay together
3. **MNI registration is mandatory** - Raw scans have different orientations/sizes
4. **Check GPU memory before increasing batch size** - 3D volumes are memory-intensive
5. **Use absolute paths in SLURM scripts** - Working directory may differ on compute nodes

## Testing & Validation

### Quick Model Test
```python
from dcan.inference.models import load_model
model = load_model('path/to/model.pt')
# Model should load without errors and accept (1, 1, 182, 218, 182) input
```

### Data Integrity Check
```python
import pandas as pd
df = pd.read_csv('data/regression.csv')
# Verify: No null Loes scores, scores in [0, 35], valid file paths
```

## Performance Benchmarks

- Training time: ~12-24 hours for 256 epochs on 2xA100
- Inference time: ~2-3 seconds per scan
- Target metrics: MAE < 2.5, R² > 0.85
- Best model: Model 17 (ResNet50) with R²=0.86