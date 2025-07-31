# Advanced Usage Guide

## Overview

This section covers advanced techniques for experienced users who need to customize the training process, scale to larger datasets, or achieve optimal performance through specialized approaches.

## Custom Loss Functions

### Understanding Loss Functions for Loes Score Prediction

The choice of loss function significantly impacts how your model learns to predict Loes scores. Different loss functions emphasize different aspects of the prediction task.

### Built-in Loss Options

#### Standard MSE Loss (Default)
```bash
# No additional flags needed - this is the default
```

**Characteristics:**
- Treats all prediction errors equally
- Penalizes large errors more heavily (quadratic penalty)
- Good for normally distributed scores

**Mathematical form**: `Loss = (predicted - actual)²`

#### Weighted MSE Loss
```bash
--use-weighted-loss
```

**Characteristics:**
- Automatically weights errors based on training data frequency
- Gives higher importance to rare Loes scores
- Helps with imbalanced datasets

**Weight calculation**: `weight[score] = 1 / frequency[score]`

### Implementing Custom Loss Functions

To implement additional loss functions, modify the `TrainingLoop` class:

#### Huber Loss (Robust to Outliers)
```python
def huber_loss(self, predictions, targets, delta=1.0):
    """
    Huber loss is less sensitive to outliers than MSE
    """
    residual = torch.abs(predictions - targets)
    condition = residual < delta
    squared_loss = 0.5 * residual ** 2
    linear_loss = delta * residual - 0.5 * delta ** 2
    return torch.where(condition, squared_loss, linear_loss)
```

#### Focal Loss for Regression
```python
def focal_regression_loss(self, predictions, targets, alpha=1.0, gamma=2.0):
    """
    Focal loss adaptation for regression - focuses on hard examples
    """
    mse = (predictions - targets) ** 2
    focal_weight = alpha * torch.pow(torch.abs(predictions - targets), gamma)
    return focal_weight * mse
```

#### Quantile Loss
```python
def quantile_loss(self, predictions, targets, quantile=0.5):
    """
    Asymmetric loss function for different quantiles
    """
    errors = targets - predictions
    return torch.max(quantile * errors, (quantile - 1) * errors)
```

### Custom Loss Integration

1. **Add loss function** to `TrainingLoop` class
2. **Modify `_compute_batch_loss`** method:
```python
if self.config.loss_function == 'huber':
    loss_g = self.huber_loss(outputs_g, label_g)
elif self.config.loss_function == 'focal':
    loss_g = self.focal_regression_loss(outputs_g, label_g)
else:
    # Default MSE loss
    loss_func = nn.MSELoss(reduction='none')
    loss_g = loss_func(outputs_g, label_g)
```

3. **Add command-line argument**:
```python
self.parser.add_argument('--loss-function', default='mse', 
                        choices=['mse', 'huber', 'focal', 'quantile'],
                        help='Loss function type')
```

## Distributed Training

### Multi-GPU Training on Single Node

#### Automatic DataParallel (Current Implementation)
The existing code automatically uses multiple GPUs when available:

```python
# In ModelHandler._init_model()
if self.use_cuda and torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

**Limitations:**
- Simple data parallelism only
- Limited scaling efficiency beyond 2-4 GPUs
- Memory replication across all GPUs

#### Implementing DistributedDataParallel (Recommended)

For better multi-GPU performance, implement DDP:

```python
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed(rank, world_size):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()

class DistributedModelHandler(ModelHandler):
    def __init__(self, model_name, device, rank):
        super().__init__(model_name, True, device)
        self.model = DDP(self.model, device_ids=[rank])
```

#### Multi-Node Training Setup

**SLURM Script Example:**
```bash
#!/bin/bash
#SBATCH --job-name=loes_training
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00

# Load modules
module load pytorch/1.13.0

# Launch distributed training
srun python src/dcan/training/regression.py \
    --csv-input-file data/large_dataset.csv \
    --folder /shared/mri/data/ \
    --csv-output-file results/distributed_predictions.csv \
    --model resnet \
    --epochs 200 \
    --batch-size 64 \
    --distributed \
    "Multi-node distributed training"
```

### Distributed Training Best Practices

#### Data Loading Considerations
```python
from torch.utils.data.distributed import DistributedSampler

# Use DistributedSampler for proper data distribution
train_sampler = DistributedSampler(train_dataset, 
                                  num_replicas=world_size, 
                                  rank=rank,
                                  shuffle=True)

train_dataloader = DataLoader(train_dataset, 
                             batch_size=batch_size,
                             sampler=train_sampler,
                             num_workers=num_workers)
```

#### Gradient Synchronization
```python
# All-reduce gradients across nodes
def sync_gradients(model):
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= dist.get_world_size()
```

## Model Fine-Tuning

### Transfer Learning Approaches

#### Pre-trained Medical Imaging Models
```python
# Load pre-trained model for medical imaging
def load_pretrained_medical_model():
    # Example: Load MedicalNet or similar pre-trained model
    model = get_resnet_model()
    pretrained_weights = torch.load('medical_pretrained_resnet.pt')
    model.load_state_dict(pretrained_weights, strict=False)
    return model
```

#### Fine-tuning Strategy

**Phase 1: Freeze Early Layers**
```python
def freeze_early_layers(model, freeze_ratio=0.7):
    """Freeze first 70% of layers for initial fine-tuning"""
    total_layers = len(list(model.parameters()))
    freeze_count = int(total_layers * freeze_ratio)
    
    for i, param in enumerate(model.parameters()):
        if i < freeze_count:
            param.requires_grad = False
```

**Phase 2: Gradual Unfreezing**
```python
def unfreeze_layers_gradually(model, epoch, total_epochs):
    """Gradually unfreeze layers as training progresses"""
    unfreeze_schedule = total_epochs // 4  # Unfreeze every quarter
    
    if epoch % unfreeze_schedule == 0:
        # Unfreeze next layer group
        pass  # Implementation depends on model architecture
```

### Domain Adaptation

#### Loes Score Range Adaptation
```python
def adapt_score_range(model, source_range, target_range):
    """
    Adapt model trained on one Loes score range to another
    Useful when transferring between different disease populations
    """
    # Linear transformation of output layer
    scale_factor = target_range[1] / source_range[1]
    bias_adjustment = target_range[0] - source_range[0] * scale_factor
    
    # Modify final layer
    with torch.no_grad():
        if hasattr(model, 'fc'):  # ResNet
            model.fc.weight *= scale_factor
            model.fc.bias = model.fc.bias * scale_factor + bias_adjustment
```

#### Cross-Scanner Adaptation
```python
def scanner_adaptation_loss(predictions, targets, scanner_ids):
    """
    Additional loss term to ensure consistent predictions across scanners
    """
    # Group by scanner
    scanner_groups = {}
    for i, scanner_id in enumerate(scanner_ids):
        if scanner_id not in scanner_groups:
            scanner_groups[scanner_id] = []
        scanner_groups[scanner_id].append(i)
    
    # Compute inter-scanner consistency loss
    consistency_loss = 0
    for indices in scanner_groups.values():
        if len(indices) > 1:
            scanner_preds = predictions[indices]
            scanner_variance = torch.var(scanner_preds)
            consistency_loss += scanner_variance
    
    return consistency_loss
```

### Advanced Fine-Tuning Configurations

#### Low Learning Rate Fine-Tuning
```bash
python src/dcan/training/regression.py \
    --csv-input-file data/new_dataset.csv \
    --folder /path/to/new/data/ \
    --model resnet \
    --lr 0.00001 \
    --epochs 100 \
    --scheduler plateau \
    --model-load-path pretrained_model.pt \
    "Fine-tuning with low LR"
```

#### Differential Learning Rates
```python
def setup_differential_lr(model, base_lr=0.001):
    """
    Set different learning rates for different parts of the model
    """
    params = [
        {'params': model.features.parameters(), 'lr': base_lr * 0.1},  # Lower LR for features
        {'params': model.classifier.parameters(), 'lr': base_lr}       # Higher LR for classifier
    ]
    return torch.optim.Adam(params)
```

## Advanced Data Augmentation

### Spatial Augmentations
```python
import torchvision.transforms as transforms

class MedicalImageAugmentation:
    def __init__(self):
        self.spatial_transforms = transforms.Compose([
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(p=0.5),
        ])
    
    def __call__(self, image):
        # Apply only valid medical imaging augmentations
        # Avoid transformations that change anatomical relationships
        return self.spatial_transforms(image)
```

### Intensity Augmentations
```python
def intensity_augmentation(image, noise_std=0.1, contrast_range=(0.8, 1.2)):
    """
    Apply realistic intensity variations
    """
    # Add Gaussian noise
    noise = torch.randn_like(image) * noise_std
    augmented = image + noise
    
    # Random contrast adjustment
    contrast_factor = torch.uniform(contrast_range[0], contrast_range[1])
    augmented = augmented * contrast_factor
    
    return augmented
```

## Ensemble Methods

### Model Ensemble Training
```python
def train_ensemble(configs, data_loaders):
    """
    Train multiple models with different configurations
    """
    models = []
    for i, config in enumerate(configs):
        print(f"Training model {i+1}/{len(configs)}")
        
        # Modify random seed for diversity
        config.random_seed = 42 + i
        
        # Train individual model
        model = train_single_model(config, data_loaders)
        models.append(model)
    
    return models

def ensemble_predict(models, input_data):
    """
    Combine predictions from multiple models
    """
    predictions = []
    for model in models:
        model.eval()
        with torch.no_grad():
            pred = model(input_data)
            predictions.append(pred)
    
    # Average predictions
    ensemble_pred = torch.stack(predictions).mean(dim=0)
    return ensemble_pred
```

### Cross-Validation Training
```python
def k_fold_cross_validation(data, k=5):
    """
    Implement k-fold cross-validation for robust performance estimation
    """
    from sklearn.model_selection import KFold
    
    subjects = data['anonymized_subject_id'].unique()
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    
    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(subjects)):
        print(f"Training fold {fold+1}/{k}")
        
        train_subjects = subjects[train_idx]
        val_subjects = subjects[val_idx]
        
        # Train model for this fold
        model = train_fold(train_subjects, val_subjects, data)
        fold_results.append(model)
    
    return fold_results
```

## Advanced Monitoring and Logging

### Custom TensorBoard Metrics
```python
def log_advanced_metrics(writer, predictions, targets, epoch):
    """
    Log additional metrics beyond basic loss and correlation
    """
    # Score distribution analysis
    pred_hist = torch.histc(predictions, bins=35, min=0, max=34)
    target_hist = torch.histc(targets, bins=35, min=0, max=34)
    
    writer.add_histogram('predictions/distribution', predictions, epoch)
    writer.add_histogram('targets/distribution', targets, epoch)
    
    # Error analysis by score range
    low_scores = targets < 5
    mid_scores = (targets >= 5) & (targets < 15)
    high_scores = targets >= 15
    
    if low_scores.any():
        low_mae = torch.abs(predictions[low_scores] - targets[low_scores]).mean()
        writer.add_scalar('error/low_scores_mae', low_mae, epoch)
    
    if mid_scores.any():
        mid_mae = torch.abs(predictions[mid_scores] - targets[mid_scores]).mean()
        writer.add_scalar('error/mid_scores_mae', mid_mae, epoch)
    
    if high_scores.any():
        high_mae = torch.abs(predictions[high_scores] - targets[high_scores]).mean()
        writer.add_scalar('error/high_scores_mae', high_mae, epoch)
```

### Model Interpretability
```python
def generate_attention_maps(model, input_image):
    """
    Generate attention maps to understand what the model focuses on
    """
    # Use GradCAM or similar technique
    from pytorch_grad_cam import GradCAM
    
    target_layers = [model.layer4[-1]]  # Last ResNet layer
    cam = GradCAM(model=model, target_layers=target_layers)
    
    grayscale_cam = cam(input_tensor=input_image)
    return grayscale_cam
```

## Performance Optimization

### Mixed Precision Training
```python
from torch.cuda.amp import autocast, GradScaler

class MixedPrecisionTraining:
    def __init__(self):
        self.scaler = GradScaler()
    
    def training_step(self, model, optimizer, inputs, targets):
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
        
        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()
        
        return loss
```

### Memory Optimization
```python
def enable_memory_optimization():
    """
    Enable various memory optimization techniques
    """
    # Gradient checkpointing
    torch.backends.cudnn.benchmark = True
    
    # Memory efficient attention (if applicable)
    torch.backends.cuda.enable_flash_sdp(True)
    
    # Clear cache periodically
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

## Advanced Configuration Examples

### Research-Grade Training
```bash
python src/dcan/training/regression.py \
    --csv-input-file data/research_dataset.csv \
    --folder /data/preprocessed_mri/ \
    --csv-output-file results/research_predictions.csv \
    --model resnet \
    --lr 0.001 \
    --batch-size 32 \
    --epochs 500 \
    --scheduler onecycle \
    --optimizer Adam \
    --use-weighted-loss \
    --num-workers 16 \
    --split-strategy stratified \
    --train-split 0.85 \
    --random-seed 42 \
    "Research-grade training with stratified split"
```

### Production Model Training
```bash
python src/dcan/training/regression.py \
    --csv-input-file data/production_dataset.csv \
    --folder /data/clinical_mri/ \
    --csv-output-file models/production_predictions.csv \
    --model resnet \
    --lr 0.0005 \
    --batch-size 64 \
    --epochs 300 \
    --scheduler plateau \
    --optimizer Adam \
    --use-weighted-loss \
    --num-workers 32 \
    --model-save-location models/production_model_v1.pt \
    --plot-location reports/production_correlation.png \
    "Production model v1 - final training"
```

### Cross-Scanner Validation
```bash
# Train on Scanner A data
python src/dcan/training/regression.py \
    --csv-input-file data/scanner_a_data.csv \
    --folder /data/scanner_a/ \
    --model-save-location models/scanner_a_model.pt \
    "Scanner A training"

# Fine-tune on Scanner B data
python src/dcan/training/regression.py \
    --csv-input-file data/scanner_b_data.csv \
    --folder /data/scanner_b/ \
    --model-load-path models/scanner_a_model.pt \
    --lr 0.0001 \
    --epochs 50 \
    --model-save-location models/cross_scanner_model.pt \
    "Cross-scanner fine-tuning"
```

## Advanced Evaluation Techniques

### Statistical Significance Testing
```python
def advanced_statistical_analysis(actual_scores, predicted_scores):
    """
    Comprehensive statistical analysis of model performance
    """
    from scipy import stats
    import numpy as np
    
    # Multiple correlation measures
    pearson_r, pearson_p = stats.pearsonr(actual_scores, predicted_scores)
    spearman_r, spearman_p = stats.spearmanr(actual_scores, predicted_scores)
    kendall_tau, kendall_p = stats.kendalltau(actual_scores, predicted_scores)
    
    # Regression metrics
    mae = np.mean(np.abs(actual_scores - predicted_scores))
    rmse = np.sqrt(np.mean((actual_scores - predicted_scores) ** 2))
    
    # Agreement analysis
    bland_altman_bias = np.mean(predicted_scores - actual_scores)
    bland_altman_limits = np.std(predicted_scores - actual_scores) * 1.96
    
    # Clinical significance thresholds
    clinically_significant_error = np.mean(np.abs(actual_scores - predicted_scores) > 2.0)
    
    return {
        'pearson_r': pearson_r, 'pearson_p': pearson_p,
        'spearman_r': spearman_r, 'spearman_p': spearman_p,
        'kendall_tau': kendall_tau, 'kendall_p': kendall_p,
        'mae': mae, 'rmse': rmse,
        'bland_altman_bias': bland_altman_bias,
        'bland_altman_limits': bland_altman_limits,
        'clinical_error_rate': clinically_significant_error
    }
```

### Cross-Dataset Validation
```python
def cross_dataset_validation(model_path, test_datasets):
    """
    Evaluate model performance across different datasets/populations
    """
    model = load_model(model_path)
    results = {}
    
    for dataset_name, dataset_path in test_datasets.items():
        test_data = load_test_data(dataset_path)
        predictions = model.predict(test_data)
        
        metrics = compute_metrics(test_data.targets, predictions)
        results[dataset_name] = metrics
        
        print(f"Dataset: {dataset_name}")
        print(f"  Correlation: {metrics['correlation']:.3f}")
        print(f"  RMSE: {metrics['rmse']:.3f}")
        print(f"  Clinical Error Rate: {metrics['clinical_error_rate']:.1%}")
    
    return results
```

## Troubleshooting Advanced Issues

### Gradient Flow Problems
```python
def check_gradient_flow(model):
    """
    Monitor gradient flow through the network
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2)
            print(f"{name}: grad_norm = {grad_norm:.6f}")
        else:
            print(f"{name}: No gradient")
```

### Memory Profiling
```python
def profile_memory_usage():
    """
    Profile GPU memory usage during training
    """
    if torch.cuda.is_available():
        print(f"GPU Memory - Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        print(f"GPU Memory - Reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB")
        print(f"GPU Memory - Max Allocated: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
```

### Model Surgery
```python
def modify_model_architecture(model, new_output_dim):
    """
    Dynamically modify model architecture
    """
    # Example: Change final layer dimension
    if hasattr(model, 'fc'):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, new_output_dim)
    
    # Re-initialize new layers
    nn.init.xavier_uniform_(model.fc.weight)
    nn.init.zeros_(model.fc.bias)
    
    return model
```

## Integration with Clinical Workflows

### DICOM Integration
```python
def process_dicom_to_model_input(dicom_path):
    """
    Convert DICOM series to model-ready format
    """
    # Convert DICOM to NIFTI
    # Apply same preprocessing pipeline as training data
    # Return model-ready tensor
    pass
```

### Real-time Inference Pipeline
```python
class ProductionInference:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)
        self.model.eval()
    
    def predict_loes_score(self, mri_path):
        """
        Production inference pipeline
        """
        # Load and preprocess image
        image = self.load_and_preprocess(mri_path)
        
        # Make prediction
        with torch.no_grad():
            prediction = self.model(image.unsqueeze(0))
        
        return prediction.item()
    
    def batch_predict(self, mri_paths):
        """
        Efficient batch prediction for multiple scans
        """
        images = [self.load_and_preprocess(path) for path in mri_paths]
        batch = torch.stack(images)
        
        with torch.no_grad():
            predictions = self.model(batch)
        
        return predictions.cpu().numpy()
```