# Early detection of cerebral ALD

This documents refers to [logistic_regression.py](/src/dcan/training/logistic_regression.py).

This is a comprehensive deep learning pipeline for **binary classification of ALD (Adrenoleukodystrophy) from MRI scans**, which builds upon and extends the previous Loes scoring system. While the first code predicted continuous Loes scores, this code classifies whether a patient **has ALD or not** based on their MRI data.

## What it does:

The system trains neural networks to automatically detect ALD (a rare genetic brain disorder) from MRI scans, essentially automating the diagnostic process that would normally require expert radiologists.


## How to run it:

### First simple run

1. `ssh -Y agate`
2. Change directory to the *scripts/training/regression* folder.  For example:
    
        cd /users/9/reine097/projects/loes-scoring-2/scripts/training/logistic-regression

3. You will be running the [create_early_detection_model.sh](../scripts/training/logistic-regression/create_early_detection_model.sh) file.  This generates the best model found so far.

        sbatch create_early_detection_model.sh 

4. You should get results similar to this:

* standardized_rmse: CHANGEME
* correlation_coefficient: CHANGEME
* Pearson correlation p-value: CHANGEME
* Spearman correlation p-value: CHANGEME

## How it works:

### **1. Core Architecture & Classes**

**`Config`**: Extensive command-line argument handling with validation for:
- Model types (simple MLP, convolutional, ResNet3D, DenseNet3D, EfficientNet3D)
- Training parameters (batch size, learning rate, epochs, optimizers)
- Data augmentation and threshold optimization settings

**`DataHandler`**: Manages data loading with support for:
- Augmented datasets for minority class balancing
- Multiple dataset formats and MRI folder structures
- Robust error handling for missing subjects/files

**`SimpleMRIModel`**: A fallback Multi-Layer Perceptron that:
- Auto-initializes input dimensions on first forward pass
- Uses dropout for regularization
- Outputs sigmoid probabilities for binary classification

### **2. Advanced Training Features**

**Threshold Optimization**: Unlike standard classification, this system:
```python
def find_optimal_threshold_for_pauc(self, val_dl, max_fpr=0.1):
```
- Finds optimal classification thresholds that maximize **partial AUC (pAUC)**
- Focuses on low false positive rates (≤10%) which is crucial for medical diagnosis
- Updates the threshold after each epoch based on validation performance

**Weighted Loss Function**: 
```python
def weighted_mse_loss(self, predictions, targets):
```
- Addresses class imbalance by weighting rare cases more heavily
- Dynamically adjusts weights based on batch composition
- Prevents the model from ignoring minority class (ALD-positive cases)

### **3. Cross-Validation with ROC Analysis**

The system implements sophisticated **k-fold cross-validation** with:

**Stratified Subject Splitting**: Ensures both ALD-positive and ALD-negative subjects are proportionally distributed across folds

**Combined ROC Curve Analysis**:
```python
def run_cross_validation_with_combined_roc(self, k=5):
```
- Plots individual ROC curves for each fold
- Calculates mean ROC curve with confidence intervals
- Generates both individual and overall performance metrics

**Comprehensive Plotting**:
- Individual fold ROC curves with different colors
- Mean ROC curve with standard deviation bands
- Overall ROC curve using all predictions combined
- Threshold optimization plots showing evolution across epochs

### **4. Medical-Specific Optimizations**

**Partial AUC (pAUC) Focus**: 
- Prioritizes performance at low false positive rates
- More clinically relevant than standard AUC for screening applications
- Helps ensure the system doesn't generate too many false alarms

**Multi-Model Architecture Support**:
- Simple MLP for baseline testing
- Convolutional networks for spatial feature extraction
- Advanced 3D models (ResNet3D, DenseNet3D) for volumetric MRI analysis

**Data Augmentation for Minority Classes**:
```python
if not is_val_set and hasattr(self, 'augment_minority') and self.augment_minority:
    dataset = AugmentedLoesScoreDataset(...)
```

### **5. Evaluation & Output**

**Comprehensive Metrics**:
- Standard classification metrics (accuracy, precision, recall, F1)
- Medical-specific metrics (sensitivity, specificity, PPV, NPV)
- ROC analysis with both standard AUC and partial AUC

**Visual Analysis**:
- ROC curves for each fold and combined analysis
- Threshold optimization evolution plots
- Comprehensive analysis plots combining multiple metrics

**Clinical Relevance**:
- Saves optimal thresholds and model weights for deployment
- Provides confidence intervals for performance estimates
- Generates publication-quality plots for medical literature

### **6. Key Differences from Loes Scoring Code**

| Aspect | Loes Scoring (Code 1) | ALD Classification (Code 2) |
|--------|----------------------|----------------------------|
| **Task** | Regression (continuous scores) | Binary classification |
| **Loss** | MSE with optional weighting | Binary Cross-Entropy with dynamic weighting |
| **Evaluation** | RMSE, correlation | ROC/AUC, sensitivity/specificity |
| **Threshold** | Fixed scoring scale | Optimized for pAUC |
| **Cross-validation** | Basic training/validation | Stratified k-fold with ROC analysis |

This system represents a more sophisticated approach to medical AI, focusing on the binary diagnostic decision (ALD vs. no ALD) rather than continuous severity scoring, with extensive validation and clinical-relevant optimization strategies.
