# Training Configuration Guide

## Overview

This section provides guidance on configuring the training process for optimal Loes score prediction performance. Proper configuration significantly impacts model accuracy, training time, and resource usage.

## Model Selection

### Available Architectures

#### ResNet (Recommended)
```bash
--model resnet
```
**Best for:**
- General-purpose Loes scoring
- Balanced accuracy and training speed
- Most datasets and hardware configurations

**Characteristics:**
- Deep residual connections handle complex brain patterns
- More robust to overfitting
- Better gradient flow during training
- Proven performance on medical imaging tasks

#### AlexNet3D
```bash
--model alexnet
```
**Best for:**
- Smaller datasets (<100 subjects)
- Limited computational resources
- Faster prototyping and experimentation

**Characteristics:**
- Simpler architecture with fewer parameters
- Faster training but potentially lower accuracy
- May struggle with complex anatomical patterns

### Model Selection Guidelines

| Dataset Size | Hardware | Recommended Model | Rationale |
|-------------|----------|------------------|-----------|
| <100 subjects | Limited GPU | AlexNet3D | Faster training, less overfitting risk |
| 100-500 subjects | Standard GPU | ResNet | Best balance of accuracy and efficiency |
| 500+ subjects | High-end GPU | ResNet | Full model capacity needed |

## Hyperparameter Configuration

### Learning Rate

#### Recommended Values
```bash
--lr 0.001    # Default, good starting point
--lr 0.0001   # Conservative, for fine-tuning
--lr 0.01     # Aggressive, for large datasets
```

#### Selection Guidelines
- **Small datasets (<100 subjects)**: Start with `0.0001`
- **Medium datasets (100-500 subjects)**: Use default `0.001`
- **Large datasets (500+ subjects)**: Try `0.01` with careful monitoring

#### Learning Rate Scheduling
```bash
--scheduler plateau   # Default: Reduce LR when validation loss plateaus
--scheduler step      # Reduce LR at fixed intervals
--scheduler cosine    # Cosine annealing schedule
--scheduler onecycle  # One-cycle learning rate policy
```

**Scheduler Comparison:**

| Scheduler | Best For | Pros | Cons |
|-----------|----------|------|------|
| `plateau` | General use, stable training | Adaptive, prevents overfitting | May be too conservative |
| `step` | Predictable training schedule | Simple, reliable | Fixed schedule may not be optimal |
| `cosine` | Long training runs | Smooth LR decay | Requires tuning T_max parameter |
| `onecycle` | Fast convergence | Often achieves best results | Requires careful total_steps calculation |

### Batch Size

#### Hardware-Based Recommendations
```bash
--batch-size 8     # 8GB GPU memory
--batch-size 16    # 16GB GPU memory  
--batch-size 32    # 24GB+ GPU memory (default)
--batch-size 64    # 48GB+ GPU memory
```

#### Considerations
- **Larger batch sizes**: More stable gradients, better GPU utilization
- **Smaller batch sizes**: More gradient updates, may generalize better
- **Memory constraints**: Reduce if encountering CUDA out-of-memory errors

### Training Duration

#### Epoch Recommendations
```bash
--epochs 50    # Quick testing/prototyping
--epochs 100   # Standard training
--epochs 200   # Thorough training for best results
--epochs 500   # Extensive training for large datasets
```

#### Early Stopping Indicators
Monitor these metrics to determine when to stop:
- Validation loss stops decreasing for 20+ epochs
- Correlation coefficient plateaus
- Training loss much lower than validation loss (overfitting)

### Optimization

#### Optimizer Selection
```bash
--optimizer Adam    # Default: Adaptive learning rates
--optimizer SGD     # Traditional: Requires careful LR tuning
```

**Adam vs SGD:**
- **Adam**: More forgiving, good default choice, handles sparse gradients well
- **SGD**: May achieve better final performance with proper tuning, more stable

## Hardware Requirements

### Minimum Specifications
- **GPU**: 8GB VRAM (GTX 1080, RTX 3070, or equivalent)
- **RAM**: 16GB system memory
- **Storage**: 100GB+ free space for data and outputs
- **CPU**: 8+ cores recommended for data loading

### Recommended Specifications
- **GPU**: 24GB+ VRAM (RTX 3090, RTX 4090, A100)
- **RAM**: 32GB+ system memory
- **Storage**: 500GB+ NVMe SSD
- **CPU**: 16+ cores for optimal data pipeline

### Multi-GPU Support
```bash
# Automatically detected and used if available
# Batch size automatically scaled by number of GPUs
```

## Advanced Configuration Options

### Loss Function Configuration

#### Standard MSE Loss (Default)
```bash
# No additional flags needed
```
- Treats all prediction errors equally
- Good for balanced datasets

#### Weighted Loss
```bash
--use-weighted-loss
```
- Gives higher weight to less frequent Loes scores
- Recommended for imbalanced datasets
- Automatically calculated based on training data distribution

### Data Loading Optimization

#### Worker Processes
```bash
--num-workers 8    # Default: Balance between speed and memory
--num-workers 4    # Conservative: Lower memory usage
--num-workers 16   # Aggressive: Maximum loading speed
```

**Guidelines:**
- Start with CPU core count
- Reduce if memory issues occur
- Increase if CPU usage is low during training

### Contrast Enhancement Handling

#### Include All Scans (Default)
```bash
# No additional flags
```

#### Exclude Gadolinium-Enhanced Scans
```bash
--gd 0
```
- Removes all contrast-enhanced scans from training
- Use when focusing on non-contrast imaging only

## Configuration Examples

### Basic Configuration (Recommended Starting Point)
```bash
python src/dcan/training/regression.py \
    --csv-input-file data/loes_data.csv \
    --folder /path/to/mri/files/ \
    --csv-output-file results/predictions.csv \
    --plot-location results/correlation_plot.png \
    --model resnet \
    --lr 0.001 \
    --batch-size 16 \
    --epochs 100 \
    --scheduler plateau \
    --optimizer Adam \
    "Basic ResNet training"
```

### High-Performance Configuration
```bash
python src/dcan/training/regression.py \
    --csv-input-file data/loes_data.csv \
    --folder /path/to/mri/files/ \
    --csv-output-file results/predictions.csv \
    --plot-location results/correlation_plot.png \
    --model resnet \
    --lr 0.01 \
    --batch-size 32 \
    --epochs 200 \
    --scheduler onecycle \
    --optimizer Adam \
    --num-workers 16 \
    "High-performance training"
```

### Memory-Constrained Configuration
```bash
python src/dcan/training/regression.py \
    --csv-input-file data/loes_data.csv \
    --folder /path/to/mri/files/ \
    --csv-output-file results/predictions.csv \
    --plot-location results/correlation_plot.png \
    --model alexnet \
    --lr 0.0001 \
    --batch-size 8 \
    --epochs 150 \
    --scheduler plateau \
    --optimizer Adam \
    --num-workers 4 \
    "Memory-efficient training"
```

### Imbalanced Dataset Configuration
```bash
python src/dcan/training/regression.py \
    --csv-input-file data/loes_data.csv \
    --folder /path/to/mri/files/ \
    --csv-output-file results/predictions.csv \
    --plot-location results/correlation_plot.png \
    --model resnet \
    --lr 0.001 \
    --batch-size 16 \
    --epochs 200 \
    --scheduler plateau \
    --optimizer Adam \
    --use-weighted-loss \
    --split-strategy stratified \
    "Weighted loss for imbalanced data"
```

## Hyperparameter Tuning Strategy

### Phase 1: Initial Validation
1. Start with basic configuration
2. Train for 50 epochs
3. Check if model is learning (correlation > 0.3)
4. Verify no obvious overfitting

### Phase 2: Learning Rate Optimization
1. Try learning rates: `[0.0001, 0.001, 0.01]`
2. Train for 100 epochs each
3. Select LR with best validation correlation
4. Consider learning rate scheduling

### Phase 3: Architecture and Duration
1. Compare ResNet vs AlexNet3D with optimal LR
2. Extend training to 200+ epochs
3. Experiment with different schedulers
4. Test weighted loss if data is imbalanced

### Phase 4: Fine-Tuning
1. Optimize batch size for your hardware
2. Adjust number of workers for data loading
3. Fine-tune scheduler parameters
4. Consider ensemble methods

## Performance Expectations

### Target Metrics
- **Correlation Coefficient**: >0.75 (excellent), >0.6 (good), >0.4 (acceptable)
- **Standardized RMSE**: <1.5 (excellent), <2.0 (good), <3.0 (acceptable)
- **Pearson p-value**: <0.001 (highly significant)

### Training Time Estimates

| Configuration | Dataset Size | Hardware | Expected Time |
|--------------|--------------|----------|---------------|
| Basic | 100 subjects | RTX 3070 | 2-4 hours |
| Standard | 300 subjects | RTX 3090 | 4-8 hours |
| High-performance | 500+ subjects | A100 | 6-12 hours |

### Convergence Indicators
- Validation loss decreases steadily for first 20-50 epochs
- Correlation coefficient improves throughout training
- Training and validation metrics don't diverge significantly

## Troubleshooting Configuration Issues

### Poor Performance (Low Correlation)
**Possible causes and solutions:**
- **Insufficient training**: Increase `--epochs`
- **Learning rate too high/low**: Try different `--lr` values
- **Data quality issues**: Review preprocessing and quality control
- **Imbalanced data**: Use `--use-weighted-loss`

### Overfitting (Training >> Validation Performance)
**Solutions:**
- Reduce learning rate
- Use plateau scheduler
- Increase validation set size
- Reduce model complexity (try AlexNet)

### Slow Training
**Optimizations:**
- Increase `--batch-size` if memory allows
- Increase `--num-workers` for faster data loading
- Use multiple GPUs if available
- Optimize data preprocessing pipeline

### Memory Issues
**Solutions:**
- Reduce `--batch-size`
- Reduce `--num-workers`
- Use `--DEBUG` flag to run on CPU for testing
- Close other GPU applications