"""
Unit tests for training utilities and pipeline.
Tests training loops, optimizers, and checkpointing.
"""

import unittest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tempfile
import os
import sys
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))


class TestTrainingLoop(unittest.TestCase):
    """Test suite for training loop components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        
        # Create simple model for testing
        self.model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        # Create fake data
        self.batch_size = 4
        self.fake_input = torch.randn(self.batch_size, 10)
        self.fake_target = torch.randn(self.batch_size, 1)
    
    def test_forward_pass(self):
        """Test forward pass through model."""
        output = self.model(self.fake_input)
        
        self.assertEqual(output.shape, (self.batch_size, 1))
        self.assertFalse(torch.any(torch.isnan(output)))
    
    def test_loss_computation(self):
        """Test loss computation."""
        output = self.model(self.fake_input)
        loss = self.criterion(output, self.fake_target)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, ())  # Scalar
        self.assertGreaterEqual(loss.item(), 0)  # MSE is non-negative
    
    def test_backward_pass(self):
        """Test backward pass and gradient computation."""
        self.model.zero_grad()
        
        output = self.model(self.fake_input)
        loss = self.criterion(output, self.fake_target)
        loss.backward()
        
        # Check that gradients are computed
        for param in self.model.parameters():
            self.assertIsNotNone(param.grad)
            self.assertFalse(torch.all(param.grad == 0))
    
    def test_optimizer_step(self):
        """Test optimizer parameter update."""
        # Store initial parameters
        initial_params = [p.clone() for p in self.model.parameters()]
        
        # Perform training step
        self.model.zero_grad()
        output = self.model(self.fake_input)
        loss = self.criterion(output, self.fake_target)
        loss.backward()
        self.optimizer.step()
        
        # Check that parameters changed
        for initial, current in zip(initial_params, self.model.parameters()):
            self.assertFalse(torch.allclose(initial, current))
    
    def test_training_mode_switch(self):
        """Test switching between train and eval modes."""
        self.model.train()
        self.assertTrue(self.model.training)
        
        self.model.eval()
        self.assertFalse(self.model.training)
    
    def test_gradient_accumulation(self):
        """Test gradient accumulation over multiple batches."""
        accumulation_steps = 4
        
        self.model.zero_grad()
        accumulated_loss = 0
        
        for step in range(accumulation_steps):
            output = self.model(self.fake_input)
            loss = self.criterion(output, self.fake_target)
            loss = loss / accumulation_steps  # Scale loss
            loss.backward()
            accumulated_loss += loss.item()
        
        # Gradients should be accumulated
        for param in self.model.parameters():
            self.assertIsNotNone(param.grad)
        
        self.optimizer.step()
        self.assertGreater(accumulated_loss, 0)


class TestLearningRateScheduler(unittest.TestCase):
    """Test learning rate scheduling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = nn.Linear(10, 1)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
    
    def test_step_lr_scheduler(self):
        """Test StepLR scheduler."""
        scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=10, 
            gamma=0.1
        )
        
        initial_lr = self.optimizer.param_groups[0]['lr']
        
        # Step through epochs
        for epoch in range(20):
            scheduler.step()
            
            if epoch == 9:  # After 10 steps
                current_lr = self.optimizer.param_groups[0]['lr']
                self.assertAlmostEqual(current_lr, initial_lr * 0.1)
    
    def test_cosine_annealing_scheduler(self):
        """Test CosineAnnealingLR scheduler."""
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=10,
            eta_min=0.0001
        )
        
        initial_lr = self.optimizer.param_groups[0]['lr']
        
        # Step through half cycle
        for _ in range(5):
            scheduler.step()
        
        current_lr = self.optimizer.param_groups[0]['lr']
        self.assertLess(current_lr, initial_lr)
        self.assertGreater(current_lr, 0.0001)
    
    def test_reduce_on_plateau_scheduler(self):
        """Test ReduceLROnPlateau scheduler."""
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=2
        )
        
        initial_lr = self.optimizer.param_groups[0]['lr']
        
        # Simulate no improvement for several epochs
        for epoch in range(5):
            val_loss = 1.0  # Constant loss (no improvement)
            scheduler.step(val_loss)
        
        current_lr = self.optimizer.param_groups[0]['lr']
        self.assertLess(current_lr, initial_lr)


class TestCheckpointing(unittest.TestCase):
    """Test model checkpointing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = nn.Linear(10, 1)
        self.optimizer = optim.Adam(self.model.parameters())
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_checkpoint(self):
        """Test saving model checkpoint."""
        checkpoint_path = os.path.join(self.temp_dir, 'checkpoint.pt')
        
        checkpoint = {
            'epoch': 10,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': 0.5,
            'best_score': 0.85
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        self.assertTrue(os.path.exists(checkpoint_path))
        self.assertGreater(os.path.getsize(checkpoint_path), 0)
    
    def test_load_checkpoint(self):
        """Test loading model checkpoint."""
        checkpoint_path = os.path.join(self.temp_dir, 'checkpoint.pt')
        
        # Save checkpoint
        original_checkpoint = {
            'epoch': 10,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': 0.5,
            'best_score': 0.85
        }
        torch.save(original_checkpoint, checkpoint_path)
        
        # Create new model and optimizer
        new_model = nn.Linear(10, 1)
        new_optimizer = optim.Adam(new_model.parameters())
        
        # Load checkpoint
        loaded_checkpoint = torch.load(checkpoint_path)
        new_model.load_state_dict(loaded_checkpoint['model_state_dict'])
        new_optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
        
        # Verify loaded values
        self.assertEqual(loaded_checkpoint['epoch'], 10)
        self.assertEqual(loaded_checkpoint['loss'], 0.5)
        self.assertEqual(loaded_checkpoint['best_score'], 0.85)
    
    def test_best_model_tracking(self):
        """Test tracking and saving best model."""
        best_score = float('inf')
        best_epoch = -1
        
        scores = [1.0, 0.8, 0.6, 0.7, 0.5, 0.6]  # Simulated validation scores
        
        for epoch, score in enumerate(scores):
            if score < best_score:
                best_score = score
                best_epoch = epoch
                
                # Save best model
                best_path = os.path.join(self.temp_dir, 'best_model.pt')
                torch.save({
                    'epoch': epoch,
                    'score': score,
                    'model_state_dict': self.model.state_dict()
                }, best_path)
        
        self.assertEqual(best_score, 0.5)
        self.assertEqual(best_epoch, 4)
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'best_model.pt')))


class TestEarlyStopping(unittest.TestCase):
    """Test early stopping functionality."""
    
    def test_early_stopping_patience(self):
        """Test early stopping with patience."""
        patience = 3
        best_score = float('inf')
        patience_counter = 0
        should_stop = False
        
        scores = [1.0, 0.8, 0.6, 0.7, 0.8, 0.9, 1.0]  # Worsening after epoch 2
        
        for epoch, score in enumerate(scores):
            if score < best_score:
                best_score = score
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                should_stop = True
                break
        
        self.assertTrue(should_stop)
        self.assertEqual(epoch, 5)  # Should stop at epoch 5
    
    def test_early_stopping_improvement(self):
        """Test early stopping resets on improvement."""
        patience = 3
        best_score = float('inf')
        patience_counter = 0
        
        scores = [1.0, 0.8, 0.9, 0.7, 0.6]  # Improvement at epochs 3 and 4
        
        for score in scores:
            if score < best_score:
                best_score = score
                patience_counter = 0
            else:
                patience_counter += 1
        
        self.assertEqual(patience_counter, 0)  # Reset after last improvement
        self.assertEqual(best_score, 0.6)


class TestMetricsTracking(unittest.TestCase):
    """Test metrics tracking during training."""
    
    def test_running_average(self):
        """Test computing running average of metrics."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        running_sum = 0
        count = 0
        
        for val in values:
            running_sum += val
            count += 1
            running_avg = running_sum / count
        
        self.assertEqual(running_avg, 3.0)
    
    def test_epoch_metrics_aggregation(self):
        """Test aggregating metrics over an epoch."""
        batch_losses = [0.5, 0.4, 0.6, 0.3, 0.2]
        batch_sizes = [32, 32, 32, 32, 16]  # Last batch is smaller
        
        total_loss = sum(loss * size for loss, size in zip(batch_losses, batch_sizes))
        total_samples = sum(batch_sizes)
        epoch_loss = total_loss / total_samples
        
        self.assertAlmostEqual(epoch_loss, 0.4133, places=3)
    
    def test_metrics_logging(self):
        """Test metrics logging structure."""
        metrics_log = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': [],
            'lr': []
        }
        
        # Simulate logging for 3 epochs
        for epoch in range(3):
            metrics_log['epoch'].append(epoch)
            metrics_log['train_loss'].append(1.0 - epoch * 0.1)
            metrics_log['val_loss'].append(0.9 - epoch * 0.08)
            metrics_log['train_mae'].append(2.0 - epoch * 0.2)
            metrics_log['val_mae'].append(1.8 - epoch * 0.15)
            metrics_log['lr'].append(0.001 * (0.9 ** epoch))
        
        # Verify structure
        self.assertEqual(len(metrics_log['epoch']), 3)
        self.assertEqual(metrics_log['train_loss'], [1.0, 0.9, 0.8])
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(metrics_log)
        self.assertEqual(len(df), 3)
        self.assertIn('val_mae', df.columns)


class TestMixedPrecisionTraining(unittest.TestCase):
    """Test mixed precision training components."""
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_amp_gradient_scaling(self):
        """Test automatic mixed precision gradient scaling."""
        model = nn.Linear(10, 1).cuda()
        optimizer = optim.Adam(model.parameters())
        scaler = torch.cuda.amp.GradScaler()
        
        input_data = torch.randn(4, 10).cuda()
        target = torch.randn(4, 1).cuda()
        
        # Forward pass with autocast
        with torch.cuda.amp.autocast():
            output = model(input_data)
            loss = nn.MSELoss()(output, target)
        
        # Backward pass with scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        self.assertIsNotNone(loss)
    
    def test_fp16_memory_savings(self):
        """Test that FP16 uses less memory than FP32."""
        size = (100, 100, 100)
        
        tensor_fp32 = torch.randn(size, dtype=torch.float32)
        tensor_fp16 = torch.randn(size, dtype=torch.float16)
        
        memory_fp32 = tensor_fp32.element_size() * tensor_fp32.nelement()
        memory_fp16 = tensor_fp16.element_size() * tensor_fp16.nelement()
        
        self.assertEqual(memory_fp16, memory_fp32 // 2)


if __name__ == '__main__':
    unittest.main()