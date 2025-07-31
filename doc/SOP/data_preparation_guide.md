# Data Preparation Guide

## Overview

Proper data preparation is critical for successful Loes score model training. This guide covers the required file formats, directory structure, data quality requirements, and best practices to ensure your model trains effectively and produces reliable predictions.

## Required Data Components

### 1. CSV Metadata File

#### Format Requirements
Your CSV file must contain the following columns in this exact format:

```csv
anonymized_subject_id,anonymized_session_id,scan,loes-score,Gd-enhanced
subject-00,session-00,mprage.nii.gz,1.0,0
subject-00,session-01,mprage.nii.gz,0.0,0
subject-00,session-02,mprage.nii.gz,1.0,0
subject-01,session-00,mprage.nii.gz,6.5,0
subject-01,session-01,mprage.nii.gz,15.0,0
```

#### Column Specifications

| Column | Type | Description | Valid Values |
|--------|------|-------------|--------------|
| `anonymized_subject_id` | String | Unique patient identifier | `subject-XX` format recommended |
| `anonymized_session_id` | String | Unique session identifier | `session-XX` format recommended |
| `scan` | String | MRI filename | Must end in `.nii.gz` |
| `loes-score` | Float | Ground truth Loes score | 0.0 to 34.0 (including decimals) |
| `Gd-enhanced` | Integer | Contrast enhancement flag | 0 (no contrast) or 1 (contrast) |

#### Optional Columns for Advanced Usage
```csv
training,validation
1,0
1,0
0,1
```
- `training`: 1 if subject should be in training set, 0 otherwise
- `validation`: 1 if subject should be in validation set, 0 otherwise

### 2. MRI File Organization

#### Directory Structure
Organize all NIFTI files in a single directory:

```
/your/data/folder/
├── subject-00_session-00_space-MNI_brain_mprage_RAVEL.nii.gz
├── subject-00_session-01_space-MNI_brain_mprage_RAVEL.nii.gz
├── subject-00_session-02_space-MNI_brain_mprage_RAVEL.nii.gz
├── subject-01_session-00_space-MNI_brain_mprage_RAVEL.nii.gz
├── subject-01_session-01_space-MNI_brain_mprage_RAVEL.nii.gz
└── ...
```

#### File Naming Convention
**Required Pattern**: `{subject_id}_{session_id}_space-MNI_brain_mprage_RAVEL.nii.gz`

**Example**: `subject-05_session-03_space-MNI_brain_mprage_RAVEL.nii.gz`

**Components**:
- `subject-05`: Must match `anonymized_subject_id` in CSV
- `session-03`: Must match `anonymized_session_id` in CSV
- `space-MNI_brain_mprage_RAVEL`: Standard preprocessing suffix
- `.nii.gz`: Compressed NIFTI format

## MRI Data Requirements

### Image Specifications
- **Format**: NIFTI (.nii.gz compressed)
- **Sequence**: T1-weighted MPRAGE or equivalent
- **Spatial Resolution**: 1mm³ isotropic recommended
- **Image Dimensions**: Typically 192×192×192 or 256×256×256

### Required Preprocessing Steps

#### 1. Skull Stripping
- Remove non-brain tissue (skull, scalp, eyes)
- Use tools like BET (FSL), 3dSkullStrip (AFNI), or FreeSurfer

#### 2. Spatial Normalization
- **Target Space**: MNI152 template
- **Registration**: Linear or non-linear registration
- **Tools**: FSL FLIRT/FNIRT, ANTs, SPM

#### 3. Intensity Normalization
- **Method**: RAVEL (Removal of Artificial Voxel Effect by Linear regression)
- **Purpose**: Standardize intensity values across subjects and scanners
- **Alternative**: Z-score normalization or histogram matching

#### 4. Quality Control
Verify each image for:
- Proper brain extraction (no skull remnants)
- Correct spatial alignment to template
- No major artifacts or distortions
- Appropriate intensity range

### Image Quality Criteria

#### Inclusion Criteria
- ✅ Complete brain coverage
- ✅ No major motion artifacts
- ✅ Proper contrast between gray/white matter
- ✅ Successful preprocessing pipeline completion

#### Exclusion Criteria
- ❌ Excessive motion artifacts
- ❌ Scanner artifacts or field inhomogeneities
- ❌ Incomplete brain coverage
- ❌ Failed preprocessing steps

## Data Split Strategy

### Critical: Prevent Data Leakage

**⚠️ ESSENTIAL RULE**: No subject can have sessions in both training and validation sets.

#### Correct Approach
```
Training Set:   subject-00 (all sessions), subject-01 (all sessions)
Validation Set: subject-02 (all sessions), subject-03 (all sessions)
```

#### Incorrect Approach (Data Leakage)
```
Training Set:   subject-00 (session-00), subject-01 (session-00)
Validation Set: subject-00 (session-01), subject-01 (session-01)  # ❌ WRONG!
```

### Split Options

#### Option 1: Automatic Runtime Split (Recommended for Beginners)
- Let the software automatically split subjects
- Specify split ratio with `--train-split 0.8` (80% training, 20% validation)
- Choose strategy: `--split-strategy random` or `stratified` or `sequential`

#### Option 2: Manual Pre-defined Split
- Add `training` and `validation` columns to your CSV
- Use `--use-train-validation-cols` flag
- Gives you full control over subject assignments

## Dataset Size Recommendations

### Minimum Requirements
- **Training**: At least 40-60 subjects
- **Validation**: At least 10-15 subjects
- **Total Sessions**: 200+ scans recommended

### Optimal Performance
- **Training**: 200+ subjects
- **Validation**: 50+ subjects
- **Total Sessions**: 1000+ scans

### Loes Score Distribution
Ensure balanced representation across score ranges:
- **Low scores (0-5)**: 30-40% of dataset
- **Medium scores (6-15)**: 40-50% of dataset  
- **High scores (16-34)**: 10-20% of dataset

## Data Validation Checklist

Before training, verify:

### File System Checks
- [ ] All NIFTI files exist in the specified folder
- [ ] File naming convention matches exactly
- [ ] No spaces or special characters in filenames
- [ ] All files are readable and not corrupted

### CSV Validation
- [ ] No missing values in required columns
- [ ] Loes scores are within valid range (0-34)
- [ ] Subject/session IDs match between CSV and filenames
- [ ] No duplicate subject-session combinations

### Data Quality
- [ ] Images are properly preprocessed
- [ ] No obvious artifacts or quality issues
- [ ] Consistent image dimensions across dataset
- [ ] Appropriate intensity ranges

### Split Validation
- [ ] No subject appears in both training and validation
- [ ] Reasonable split ratios (typically 70-80% training)
- [ ] Balanced Loes score distribution in both sets

## Common Data Preparation Issues

### File Path Problems
**Issue**: "File not found" errors during training
**Solutions**:
- Check folder path ends with `/`
- Verify exact filename matching between CSV and files
- Ensure no hidden characters or encoding issues

### Image Format Issues
**Issue**: "Cannot load NIFTI file" errors
**Solutions**:
- Verify files are valid NIFTI format
- Check file permissions are readable
- Ensure files aren't corrupted (try opening with FSL or other tools)

### Memory Issues
**Issue**: Out of memory during data loading
**Solutions**:
- Reduce batch size (`--batch-size 16` or `8`)
- Use fewer worker processes (`--num-workers 4`)
- Ensure images aren't unnecessarily large

### Score Distribution Problems
**Issue**: Poor model performance due to imbalanced data
**Solutions**:
- Use weighted loss (`--use-weighted-loss`)
- Ensure adequate representation across score ranges
- Consider data augmentation for underrepresented scores

## Data Preprocessing Pipeline Example

```bash
# 1. Skull stripping with FSL BET
bet input.nii.gz brain.nii.gz -f 0.5 -g 0

# 2. Registration to MNI space
flirt -in brain.nii.gz -ref $FSLDIR/data/standard/MNI152_T1_1mm.nii.gz \
      -out brain_mni.nii.gz -omat brain_to_mni.mat

# 3. Apply RAVEL intensity normalization
# (Use R RAVEL package or equivalent Python implementation)

# 4. Final filename
mv brain_mni_ravel.nii.gz subject-XX_session-XX_space-MNI_brain_mprage_RAVEL.nii.gz
```

## Quality Control Recommendations

### Visual Inspection
- Review 5-10% of randomly selected images
- Check spatial normalization quality
- Verify skull stripping completeness
- Confirm intensity normalization consistency

### Automated Checks
- Calculate basic image statistics (mean, std, range)
- Check for outliers in intensity distributions
- Verify image dimensions consistency
- Validate file integrity

### Documentation
- Record preprocessing software versions
- Document any excluded subjects and reasons
- Keep track of quality control metrics
- Maintain preprocessing parameter settings