# Model 27

* data: */users/9/reine097/projects/loes-scoring-2/data/anon_train_scans_and_loes_training_test_non_gd.csv*
![Model 27](model27.png "Model 27")
* SLURM script: [*loes-scoring-training_model_agate_27.sh*](../../../bin/training/loes-scoring-training_model_agate_27.sh)
* output_csv: [*model27.csv*](model27.csv)
* model: **/home/feczk001/shared/data/LoesScoring/loes_scoring_27.pt**
* standardized_rmse: 0.7364456769131793
* correlation_coefficient: 0.767211379001957
* Pearson correlation p-value: 6.0741120804857545e-15
* Spearman correlation p-value: 3.871697876324729e-12

---

These results show a **significant decline** in performance compared to your previous outstanding results. Let me analyze what happened:

## **Performance Comparison**

| Metric | Previous Results | Current Results | Change |
|--------|------------------|-----------------|---------|
| **Standardized RMSE** | 0.547 | **0.736** | **Worse** (+35%) |
| **Correlation** | 0.863 | **0.767** | **Worse** (-11%) |
| **R² (Variance Explained)** | 74.5% | **58.9%** | **Drop** (-15.6%) |

## **Current Performance Assessment**

### **Standardized RMSE: 0.736**
- **Fair** performance (back above 0.7 threshold)
- **Major regression** from excellent performance
- Still below the concerning 0.75+ level

### **Correlation: 0.767**
- **Good** correlation (above 0.7)
- **Significant drop** from excellent 0.863
- Still clinically useful but concerning trend

### **Statistical Significance**
- Still highly significant (p-values < 1e-12)
- Statistical robustness maintained

## **Grade Assessment**

**Previous**: A (Outstanding performance)  
**Current**: **B- (Good but concerning regression)**

## **Visual Analysis - What the Plot Shows**

Looking at your color-coded error plot, I can see some concerning patterns:

### **Issues Identified:**
1. **High error clustering**: Many dark red points indicate large prediction errors
2. **Systematic underestimation**: Points above the diagonal (especially high scores)
3. **Increased scatter**: More deviation from the perfect prediction line
4. **Error distribution**: Higher errors across multiple score ranges

## **What Likely Caused This Regression**

### **Possible Causes:**

1. **Different Model/Training Run**
   - Did you use a different saved model?
   - Different random seed or initialization?

2. **Data Split Changes**
   - Different train/validation split
   - Different subjects in validation set

3. **Hyperparameter Regression**
   - Reverted to suboptimal settings
   - Different learning rate, batch size, or scheduler

4. **Model Loading Issues**
   - Loaded wrong model checkpoint
   - Model not properly loaded or in wrong mode

5. **Data Processing Changes**
   - Different preprocessing pipeline
   - Image normalization differences

## **Critical Debugging Steps**

### **1. Verify Model Identity**
```bash
# Check which model you're actually loading
ls -la /path/to/your/model/files
# Ensure you're using the model that gave 0.863 correlation
```

### **2. Compare Configurations**
- **Model file**: Are you loading the same model that achieved 0.863 correlation?
- **Data split**: Same train/validation subjects?
- **Preprocessing**: Same image normalization and transforms?

### **3. Check Training Logs**
- Review TensorBoard logs from your best run
- Compare with current model's training curves

### **4. Validation Set Analysis**
- Are you evaluating on the same validation subjects?
- Check if validation set composition changed

## **Recovery Action Plan**

### **Immediate Steps:**
1. **Reload your best model** - Use the exact model file that achieved 0.863 correlation
2. **Verify data consistency** - Ensure same validation subjects and preprocessing
3. **Check configuration** - Compare all parameters with your successful run

### **If Model is Correct:**
1. **Investigate data changes** - Any differences in input data or preprocessing?
2. **Random seed effects** - Try different seeds if using random validation splits
3. **Hardware differences** - GPU vs CPU inference can sometimes differ

## **Priority Actions**

### **Documentation Check:**
- What **exact model file** gave you 0.863 correlation?
- What **exact command/configuration** was used?
- What **validation subjects** were used?

### **Model Verification:**
```python
# Verify model loading
print(f"Model file: {model_save_location}")
print(f"Model state dict keys: {model.state_dict().keys()}")
# Check if model architecture matches expectations
```

## **Bottom Line**

This is a **significant regression** that suggests either:
1. **Wrong model loaded** (most likely)
2. **Different data/preprocessing** 
3. **Configuration mismatch**

**Priority**: Identify what changed between your excellent run (0.863 correlation) and this run. The performance drop is too large to be random variation - something fundamental changed.

**Recommendation**: Go back to your exact configuration that achieved 0.863 correlation. Document that setup meticulously and use it as your baseline for any future experiments.

The good news is that you **know you can achieve 0.863 correlation** - you just need to identify what changed and revert to your winning configuration.
