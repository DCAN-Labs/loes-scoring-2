# Troubleshooting Guide

## Common Errors and Solutions

### File and Data Errors

#### Error: "FileNotFoundError" or "No such file or directory"

**Symptoms:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'subject-00_session-00_space-MNI_brain_mprage_RAVEL.nii.gz'
```

**Common Causes and Solutions:**

1. **Incorrect folder path**
   ```bash
   # Wrong - missing trailing slash
   --folder /path/to/data
   
   # Correct - include trailing slash
   --folder /path/to/data/
   ```

2. **Filename mismatch between CSV and actual files**
   - Check exact spelling and case sensitivity
   - Verify file extensions (.nii.gz vs .nii)
   - Ensure no extra spaces or hidden characters

3. **Missing files**
   ```bash
   # Verify all files exist
   ls /path/to/data/subject-*_session-*_space-MNI_brain_mprage_RAVEL.nii.gz | wc -l
   ```

#### Error: "NIFTI file cannot be loaded"

**Symptoms:**
```
Error loading NIFTI file: subject-01_session-00_space-MNI_brain_mprage_RAVEL.nii.gz
```

**Solutions:**
1. **Verify file integrity**
   ```bash
   # Test with FSL
   fslinfo subject-01_session-00_space-MNI_brain_mprage_RAVEL.nii.gz
   
   # Test with Python
   import nibabel as nib
   img = nib.load('your_file.nii.gz')
   ```

2. **Check file permissions**
   ```bash
   chmod 644 *.nii.gz
   ```

3. **Re-compress corrupted files**
   ```bash
   gunzip file.nii.gz
   gzip file.nii
   ```

#### Error: "Data leakage detected" or Poor validation performance

**Symptoms:**
- Training accuracy much higher than validation accuracy
- Unrealistically high correlation coefficients
- Model fails on external test data

**Solutions:**
1. **Verify subject-level splits**
   ```python
   # Check for overlap
   train_subjects = set(train_data['anonymized_subject_id'])
   val_subjects = set(val_data['anonymized_subject_id'])
   overlap = train_subjects.intersection(val_subjects)
   if overlap:
       print(f"Data leakage detected: {overlap}")
   ```

2. **Use proper split strategy**
   ```bash
   # Ensure subject-level splitting
   --split-strategy stratified
   --random-seed 42  # For reproducibility
   ```

### Memory and Hardware Errors

#### Error: "CUDA out of memory"

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 23.69 GiB total capacity)
```

**Immediate Solutions:**
```bash
# Reduce batch size
--batch-size 8   # From default 32

# Reduce workers
--num-workers 2  # From default 8

# Enable debug mode (CPU only)
--DEBUG
```

**Advanced Solutions:**
1. **Enable gradient checkpointing**
   ```python
   # In model initialization
   model.gradient_checkpointing = True
   ```

2. **Use mixed precision training**
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   
   with autocast():
       outputs = model(inputs)
       loss = criterion(outputs, targets)
   ```

3. **Clear cache periodically**
   ```python
   if batch_idx % 100 == 0:
       torch.cuda.empty_cache()
   ```

#### Error: "DataLoader worker killed" or Hanging data loading

**Symptoms:**
- Training hangs during data loading
- Workers crashed messages
- No progress after "Starting training..."

**Solutions:**
```bash
# Reduce number of workers
--num-workers 0   # Single-threaded loading
--num-workers 2   # Conservative multi-threading

# Check system resources
htop  # Monitor CPU and memory usage
nvidia-smi  # Monitor GPU usage
```

### Training Performance Issues

#### Problem: Very low correlation coefficient (<0.3)

**Diagnostic Steps:**
1. **Check data quality**
   ```python
   # Verify Loes score distribution
   df['loes-score'].describe()
   df['loes-score'].hist()
   
   # Check for missing values
   df.isnull().sum()
   ```

2. **Verify model is learning**
   ```bash
   # Monitor training loss
   tensorboard --logdir runs/
   ```

**Solutions:**
- Increase learning rate: `--lr 0.01`
- Extend training: `--epochs 200`
- Try different model: `--model alexnet`
- Use weighted loss: `--use-weighted-loss`

#### Problem: Training loss decreases but validation loss increases (Overfitting)

**Symptoms:**
- Training correlation > 0.8, validation correlation < 0.5
- Training loss continues decreasing while validation loss increases

**Solutions:**
1. **Reduce model complexity**
   ```bash
   --model alexnet  # Simpler architecture
   ```

2. **Regularization techniques**
   ```bash
   --lr 0.0001      # Lower learning rate
   --scheduler plateau  # Adaptive learning rate reduction
   ```

3. **Early stopping**
   - Monitor validation metrics
   - Stop when validation performance plateaus

4. **Increase validation set size**
   ```bash
   --train-split 0.7  # 70% training, 30% validation
   ```

#### Problem: Training is extremely slow

**Symptoms:**
- Very slow epoch completion times
- Low GPU utilization

**Performance Diagnostics:**
```bash
# Monitor GPU usage
nvidia-smi -l 1

# Check CPU usage
htop

# Profile data loading
python -m torch.utils.bottleneck your_training_script.py
```

**Solutions:**
1. **Optimize data loading**
   ```bash
   --num-workers 16     # Increase if CPU cores available
   --batch-size 64      # Increase if GPU memory allows
   ```

2. **Hardware optimizations**
   ```bash
   # Enable CUDA optimizations
   export CUDA_LAUNCH_BLOCKING=0
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
   ```

## Performance Optimization

### Hardware Optimization

#### GPU Optimization
```bash
# Check GPU configuration
nvidia-smi
nvidia-smi topo -m  # Check GPU topology

# Optimize CUDA settings
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Specify GPUs to use
export NCCL_DEBUG=INFO               # For distributed training debug
```

#### CPU and Memory Optimization
```bash
# Set CPU affinity for data loading workers
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Optimize memory allocation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

### Training Speed Optimization

#### Batch Size Tuning
```python
def find_optimal_batch_size(model, device):
    """
    Find largest batch size that fits in memory
    """
    batch_size = 1
    while True:
        try:
            # Create dummy batch
            dummy_input = torch.randn(batch_size, 1, 192, 192, 192).to(device)
            dummy_target = torch.randn(batch_size).to(device)
            
            # Forward pass
            output = model(dummy_input)
            loss = nn.MSELoss()(output.squeeze(), dummy_target)
            loss.backward()
            
            print(f"Batch size {batch_size}: OK")
            batch_size *= 2
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                optimal_batch_size = batch_size // 2
                print(f"Optimal batch size: {optimal_batch_size}")
                return optimal_batch_size
            else:
                raise e
```

#### Data Loading Optimization
```python
# Optimize dataset loading
class OptimizedLoesScoreDataset(LoesScoreDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = {}  # Simple caching mechanism
    
    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        
        item = super().__getitem__(idx)
        
        # Cache if memory allows
        if len(self.cache) < 100:  # Limit cache size
            self.cache[idx] = item
        
        return item
```

### Model Performance Optimization

#### Learning Rate Scheduling
```bash
# For fast convergence
--scheduler onecycle --lr 0.01

# For stable training
--scheduler plateau --lr 0.001

# For long training runs
--scheduler cosine --lr 0.001
```

#### Architecture-Specific Optimizations
```python
# ResNet optimization
def optimize_resnet(model):
    # Enable efficient attention if available
    for module in model.modules():
        if hasattr(module, 'use_efficient_attention'):
            module.use_efficient_attention = True

# AlexNet optimization  
def optimize_alexnet(model):
    # Use in-place operations where possible
    for module in model.modules():
        if hasattr(module, 'inplace'):
            module.inplace = True
```

## Debugging Tips

### Enable Debug Mode

#### Basic Debug Configuration
```bash
python src/dcan/training/regression.py \
    --DEBUG \
    --batch-size 2 \
    --epochs 1 \
    --num-workers 0 \
    --csv-input-file data/small_test.csv \
    "Debug run"
```

**Debug mode changes:**
- Forces CPU usage (no CUDA)
- Enables verbose logging
- Smaller batch sizes
- Single-threaded data loading

### Logging and Monitoring

#### Enable Detailed Logging
```python
import logging

# Set up comprehensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_debug.log'),
        logging.StreamHandler()
    ]
)

# Log model architecture
log.debug(f"Model architecture:\n{model}")

# Log tensor shapes
log.debug(f"Input shape: {input_tensor.shape}")
log.debug(f"Output shape: {output_tensor.shape}")
```

#### TensorBoard Debugging
```python
def log_debug_info(writer, epoch, model, inputs, outputs, targets):
    """
    Log detailed debugging information to TensorBoard
    """
    # Log model gradients
    for name, param in model.named_parameters():
        if param.grad is not None:
            writer.add_histogram(f'gradients/{name}', param.grad, epoch)
            writer.add_scalar(f'gradient_norms/{name}', param.grad.norm(), epoch)
    
    # Log activations
    writer.add_histogram('activations/outputs', outputs, epoch)
    writer.add_histogram('targets/distribution', targets, epoch)
    
    # Log learning rate
    writer.add_scalar('hyperparams/learning_rate', 
                     optimizer.param_groups[0]['lr'], epoch)
```

### Step-by-Step Debugging Process

#### Step 1: Verify Data Pipeline
```python
def debug_data_pipeline(csv_file, folder):
    """
    Debug the data loading pipeline
    """
    print("=== Data Pipeline Debug ===")
    
    # Check CSV loading
    df = pd.read_csv(csv_file)
    print(f"CSV shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"First few rows:\n{df.head()}")
    
    # Check file existence
    missing_files = []
    for _, row in df.iterrows():
        subject_id = row['anonymized_subject_id']
        session_id = row['anonymized_session_id']
        filename = f"{subject_id}_{session_id}_space-MNI_brain_mprage_RAVEL.nii.gz"
        filepath = os.path.join(folder, filename)
        
        if not os.path.exists(filepath):
            missing_files.append(filepath)
    
    if missing_files:
        print(f"Missing files: {len(missing_files)}")
        print("First 5 missing:", missing_files[:5])
    else:
        print("All files found!")
```

#### Step 2: Test Model Forward Pass
```python
def debug_model_forward(model, device):
    """
    Test model with dummy data
    """
    print("=== Model Forward Pass Debug ===")
    
    model.eval()
    dummy_input = torch.randn(1, 1, 192, 192, 192).to(device)
    
    try:
        with torch.no_grad():
            output = model(dummy_input)
        print(f"Forward pass successful!")
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output value: {output.item():.3f}")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Input device: {dummy_input.device}")
```

#### Step 3: Debug Training Loop
```python
def debug_training_step(model, dataloader, optimizer, device):
    """
    Debug a single training step
    """
    print("=== Training Step Debug ===")
    
    model.train()
    batch = next(iter(dataloader))
    inputs, targets, _, _ = batch
    
    print(f"Batch size: {inputs.shape[0]}")
    print(f"Input shape: {inputs.shape}")
    print(f"Target shape: {targets.shape}")
    print(f"Target values: {targets}")
    
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    # Forward pass
    optimizer.zero_grad()
    outputs = model(inputs)
    
    print(f"Model output shape: {outputs.shape}")
    print(f"Model output values: {outputs.squeeze()}")
    
    # Loss calculation
    loss_fn = nn.MSELoss()
    loss = loss_fn(outputs.squeeze(), targets)
    
    print(f"Loss: {loss.item():.6f}")
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
    print(f"Gradient norm: {grad_norm:.6f}")
    
    optimizer.step()
    print("Training step completed successfully!")
```

### CSV and Data Format Issues

#### Error: "KeyError" or Missing columns

**Symptoms:**
```
KeyError: 'loes-score'
KeyError: 'anonymized_subject_id'
```

**Solutions:**
1. **Verify exact column names**
   ```python
   import pandas as pd
   df = pd.read_csv('your_file.csv')
   print("Columns:", df.columns.tolist())
   print("Expected: ['anonymized_subject_id', 'anonymized_session_id', 'scan', 'loes-score', 'Gd-enhanced']")
   ```

2. **Check for hidden characters**
   ```python
   # Remove whitespace from column names
   df.columns = df.columns.str.strip()
   ```

3. **Standardize column names**
   ```python
   # Rename columns if needed
   df.rename(columns={
       'subject_id': 'anonymized_subject_id',
       'session_id': 'anonymized_session_id',
       'loes_score': 'loes-score'
   }, inplace=True)
   ```

#### Error: Invalid Loes score values

**Symptoms:**
```
ValueError: Loes score 45.0 outside valid range [0, 34]
```

**Solutions:**
1. **Check score range**
   ```python
   print(f"Loes score range: {df['loes-score'].min()} to {df['loes-score'].max()}")
   print(f"Invalid scores: {df[df['loes-score'] > 34]['loes-score'].tolist()}")
   ```

2. **Clean invalid data**
   ```python
   # Remove invalid scores
   df = df[(df['loes-score'] >= 0) & (df['loes-score'] <= 34)]
   ```

### Model Training Issues

#### Problem: Loss is NaN or exploding

**Symptoms:**
```
Epoch 1: Loss = nan
Epoch 1: Loss = inf
```

**Diagnostic Steps:**
1. **Check input data**
   ```python
   print(f"Input min/max: {inputs.min():.3f}/{inputs.max():.3f}")
   print(f"Target min/max: {targets.min():.3f}/{targets.max():.3f}")
   print(f"Any NaN in inputs: {torch.isnan(inputs).any()}")
   print(f"Any NaN in targets: {torch.isnan(targets).any()}")
   ```

2. **Check gradients**
   ```python
   for name, param in model.named_parameters():
       if param.grad is not None:
           grad_norm = param.grad.norm()
           print(f"{name}: grad_norm = {grad_norm:.6f}")
   ```

**Solutions:**
- Reduce learning rate: `--lr 0.0001`
- Enable gradient clipping
- Check data preprocessing for extreme values
- Use different optimizer: `--optimizer sgd`

#### Problem: Model not converging

**Symptoms:**
- Loss stays constant across epochs
- Very low correlation coefficients
- No improvement in validation metrics

**Solutions:**
1. **Increase learning rate**
   ```bash
   --lr 0.01  # From default 0.001
   ```

2. **Change optimizer**
   ```bash
   --optimizer sgd --lr 0.1
   ```

3. **Verify data preprocessing**
   - Check if images are properly normalized
   - Ensure consistent preprocessing across all images

4. **Try different architecture**
   ```bash
   --model alexnet  # Simpler model may learn faster
   ```

## Performance Optimization

### Training Speed Optimization

#### Profiling Training Performance
```python
import time
import torch.profiler

def profile_training():
    """
    Profile training performance to identify bottlenecks
    """
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for epoch in range(5):
            train_one_epoch()
            prof.step()
```

#### Data Loading Bottlenecks
```bash
# Test different worker counts
for workers in 0 2 4 8 16; do
    echo "Testing $workers workers"
    time python training.py --num-workers $workers --epochs 1
done
```

#### Memory Usage Optimization
```python
def optimize_memory_usage():
    """
    Various memory optimization techniques
    """
    # Use memory-efficient attention
    torch.backends.cuda.enable_flash_sdp(True)
    
    # Optimize memory allocation
    torch.backends.cudnn.benchmark = True
    
    # Use channels-last memory format
    model = model.to(memory_format=torch.channels_last_3d)
    inputs = inputs.to(memory_format=torch.channels_last_3d)
```

### Model Performance Optimization

#### Hyperparameter Search
```python
def automated_hyperparameter_search():
    """
    Simple grid search for optimal hyperparameters
    """
    learning_rates = [0.0001, 0.001, 0.01]
    batch_sizes = [8, 16, 32]
    schedulers = ['plateau', 'step', 'onecycle']
    
    best_config = None
    best_correlation = 0
    
    for lr in learning_rates:
        for batch_size in batch_sizes:
            for scheduler in schedulers:
                print(f"Testing: lr={lr}, batch_size={batch_size}, scheduler={scheduler}")
                
                # Train model with current configuration
                correlation = train_and_evaluate(lr, batch_size, scheduler)
                
                if correlation > best_correlation:
                    best_correlation = correlation
                    best_config = (lr, batch_size, scheduler)
    
    print(f"Best configuration: {best_config}")
    print(f"Best correlation: {best_correlation}")
```

#### Model Architecture Optimization
```python
def compare_model_architectures():
    """
    Systematically compare different model configurations
    """
    configs = [
        {'model': 'resnet', 'lr': 0.001},
        {'model': 'alexnet', 'lr': 0.001},
        {'model': 'resnet', 'lr': 0.01},
        {'model': 'alexnet', 'lr': 0.01},
    ]
    
    results = {}
    for config in configs:
        print(f"Testing configuration: {config}")
        correlation = train_model(**config)
        results[str(config)] = correlation
    
    # Sort by performance
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    print("Results (best to worst):")
    for config, correlation in sorted_results:
        print(f"  {config}: {correlation:.3f}")
```

## Debugging Tips

### Setting Up Debug Environment

#### Create Minimal Test Dataset
```python
def create_debug_dataset(original_csv, output_csv, n_subjects=5):
    """
    Create small dataset for debugging
    """
    df = pd.read_csv(original_csv)
    
    # Select first N subjects
    unique_subjects = df['anonymized_subject_id'].unique()[:n_subjects]
    debug_df = df[df['anonymized_subject_id'].isin(unique_subjects)]
    
    debug_df.to_csv(output_csv, index=False)
    print(f"Debug dataset created with {len(debug_df)} rows from {n_subjects} subjects")
```

#### Quick Validation Script
```python
def quick_validation_check():
    """
    Fast validation of entire pipeline
    """
    print("=== Quick Pipeline Validation ===")
    
    # 1. Data loading test
    debug_data_pipeline('data/debug.csv', '/path/to/data/')
    
    # 2. Model test
    model = get_resnet_model()
    device = torch.device('cpu')  # Use CPU for debugging
    model = model.to(device)
    debug_model_forward(model, device)
    
    # 3. Training step test
    from torch.utils.data import DataLoader
    dataset = LoesScoreDataset(folder, subjects, df, df)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    debug_training_step(model, dataloader, optimizer, device)
    
    print("=== All tests passed! ===")
```

### Common Debugging Commands

#### Check Environment
```bash
# Python environment
python --version
pip list | grep torch
pip list | grep pandas

# System resources
free -h          # Check memory
df -h            # Check disk space
nvidia-smi       # Check GPU status

# CUDA version compatibility
python -c "import torch; print(torch.version.cuda)"
nvidia-smi | grep "CUDA Version"
```

#### Validate Installation
```python
# Test imports
try:
    import torch
    import pandas as pd
    import nibabel as nib
    print("All imports successful")
except ImportError as e:
    print(f"Import error: {e}")

# Test CUDA
if torch.cuda.is_available():
    print(f"CUDA available: {torch.cuda.device_count()} devices")
    print(f"Current device: {torch.cuda.current_device()}")
else:
    print("CUDA not available")
```

### Advanced Debugging Techniques

#### Memory Debugging
```python
def debug_memory_usage():
    """
    Track memory usage throughout training
    """
    import psutil
    import gc
    
    process = psutil.Process()
    
    def log_memory():
        cpu_mem = process.memory_info().rss / 1024 / 1024  # MB
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            print(f"CPU Memory: {cpu_mem:.1f}MB, GPU Memory: {gpu_mem:.1f}MB")
        else:
            print(f"CPU Memory: {cpu_mem:.1f}MB")
    
    # Log memory at different points
    log_memory()  # Before training
    # ... training code ...
    log_memory()  # After training
    
    # Force garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

#### Gradient Debugging
```python
def debug_gradients(model, loss):
    """
    Detailed gradient analysis
    """
    print("=== Gradient Debug ===")
    
    # Check if gradients exist
    has_gradients = any(param.grad is not None for param in model.parameters())
    print(f"Model has gradients: {has_gradients}")
    
    if has_gradients:
        # Compute gradient statistics
        total_norm = 0
        param_count = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                
                print(f"{name}:")
                print(f"  Shape: {param.shape}")
                print(f"  Grad norm: {param_norm:.6f}")
                print(f"  Grad mean: {param.grad.mean():.6f}")
                print(f"  Grad std: {param.grad.std():.6f}")
        
        total_norm = total_norm ** (1. / 2)
        print(f"Total gradient norm: {total_norm:.6f}")
        print(f"Parameters with gradients: {param_count}")
```

### Emergency Recovery Procedures

#### Recover from Corrupted Training
```python
def recover_training(checkpoint_dir, csv_file):
    """
    Recover training from the last good checkpoint
    """
    import glob
    
    # Find latest checkpoint
    checkpoints = glob.glob(os.path.join(checkpoint_dir, 'model-*.pt'))
    if not checkpoints:
        print("No checkpoints found!")
        return None
    
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    print(f"Loading checkpoint: {latest_checkpoint}")
    
    # Load model
    model = get_resnet_model()
    model.load_state_dict(torch.load(latest_checkpoint))
    
    # Resume training from this point
    return model
```

#### Clean Up Failed Runs
```bash
#!/bin/bash
# cleanup_failed_training.sh

echo "Cleaning up failed training artifacts..."

# Remove incomplete model files
find . -name "model-*.pt" -size -1M -delete

# Clean up empty log directories
find runs/ -type d -empty -delete

# Clear CUDA cache
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"

# Reset file permissions
chmod -R 644 data/*.csv
chmod -R 644 data/*.nii.gz

echo "Cleanup completed!"
```

### Performance Benchmarking

#### Benchmark Different Configurations
```bash
#!/bin/bash
# benchmark_configs.sh

echo "Benchmarking different configurations..."

configs=(
    "--model resnet --lr 0.001 --batch-size 16"
    "--model resnet --lr 0.01 --batch-size 32" 
    "--model alexnet --lr 0.001 --batch-size 16"
    "--model alexnet --lr 0.01 --batch-size 32"
)

for config in "${configs[@]}"; do
    echo "Testing: $config"
    
    start_time=$(date +%s)
    python src/dcan/training/regression.py \
        --csv-input-file data/benchmark.csv \
        --folder /data/benchmark/ \
        --epochs 10 \
        $config \
        "Benchmark test" > "benchmark_$(echo $config | tr ' ' '_').log" 2>&1
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    echo "Configuration '$config' took $duration seconds"
done
```

#### Resource Monitoring Script
```python
def monitor_resources(duration_minutes=60):
    """
    Monitor system resources during training
    """
    import time
    import psutil
    
    end_time = time.time() + (duration_minutes * 60)
    
    with open('resource_monitor.log', 'w') as f:
        f.write("timestamp,cpu_percent,memory_percent,gpu_memory_mb\n")
        
        while time.time() < end_time:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            gpu_memory = 0
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
            
            f.write(f"{timestamp},{cpu_percent},{memory_percent},{gpu_memory}\n")
            f.flush()
            
            time.sleep(30)  # Log every 30 seconds
```