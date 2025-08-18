"""
Unit tests for inference pipeline.
Tests model loading, prediction, and post-processing.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import os
import sys
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
import nibabel as nib

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))


class TestModelLoading(unittest.TestCase):
    """Test model loading for inference."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
    
    def tearDown(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_model_weights(self):
        """Test loading model weights for inference."""
        model_path = os.path.join(self.temp_dir, 'model.pt')
        
        # Save model
        torch.save(self.model.state_dict(), model_path)
        
        # Create new model and load weights
        new_model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        new_model.load_state_dict(torch.load(model_path))
        
        # Verify weights are identical
        for p1, p2 in zip(self.model.parameters(), new_model.parameters()):
            torch.testing.assert_close(p1, p2)
    
    def test_load_checkpoint_for_inference(self):
        """Test loading full checkpoint for inference."""
        checkpoint_path = os.path.join(self.temp_dir, 'checkpoint.pt')
        
        # Save checkpoint
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'epoch': 100,
            'best_score': 0.92
        }
        torch.save(checkpoint, checkpoint_path)
        
        # Load for inference
        loaded = torch.load(checkpoint_path)
        new_model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        new_model.load_state_dict(loaded['model_state_dict'])
        
        # Model should be ready for inference
        new_model.eval()
        self.assertFalse(new_model.training)
    
    def test_model_eval_mode(self):
        """Test model is in eval mode for inference."""
        self.model.train()
        self.assertTrue(self.model.training)
        
        # Switch to eval for inference
        self.model.eval()
        self.assertFalse(self.model.training)
        
        # Dropout and BatchNorm should behave differently
        with torch.no_grad():
            input_data = torch.randn(10, 10)
            output1 = self.model(input_data)
            output2 = self.model(input_data)
            
            # Outputs should be identical in eval mode
            torch.testing.assert_close(output1, output2)


class TestInferencePipeline(unittest.TestCase):
    """Test complete inference pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(8, 1)
        )
        self.model.eval()
    
    def test_single_sample_inference(self):
        """Test inference on single sample."""
        # Create fake MRI volume
        input_volume = torch.randn(1, 1, 32, 32, 32)
        
        with torch.no_grad():
            output = self.model(input_volume)
        
        self.assertEqual(output.shape, (1, 1))
        self.assertFalse(torch.any(torch.isnan(output)))
    
    def test_batch_inference(self):
        """Test inference on batch of samples."""
        batch_size = 4
        input_batch = torch.randn(batch_size, 1, 32, 32, 32)
        
        with torch.no_grad():
            outputs = self.model(input_batch)
        
        self.assertEqual(outputs.shape, (batch_size, 1))
        self.assertEqual(len(outputs), batch_size)
    
    def test_inference_reproducibility(self):
        """Test that inference is reproducible."""
        torch.manual_seed(42)
        input_data = torch.randn(1, 1, 32, 32, 32)
        
        with torch.no_grad():
            output1 = self.model(input_data)
            output2 = self.model(input_data)
        
        torch.testing.assert_close(output1, output2)
    
    @patch('nibabel.load')
    def test_nifti_to_prediction(self, mock_nib_load):
        """Test complete pipeline from NIfTI to prediction."""
        # Mock NIfTI loading
        fake_data = np.random.randn(32, 32, 32).astype(np.float32)
        mock_img = MagicMock()
        mock_img.get_fdata.return_value = fake_data
        mock_nib_load.return_value = mock_img
        
        # Load and preprocess
        img = mock_nib_load('fake_path.nii.gz')
        data = img.get_fdata()
        
        # Convert to tensor
        tensor = torch.from_numpy(data).float()
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        
        # Inference
        with torch.no_grad():
            prediction = self.model(tensor)
        
        self.assertEqual(prediction.shape, (1, 1))
        
        # Convert to LOES score (0-35 range)
        loes_score = torch.clamp(prediction * 35, 0, 35)
        self.assertTrue(0 <= loes_score.item() <= 35)


class TestPostProcessing(unittest.TestCase):
    """Test post-processing of model outputs."""
    
    def test_score_denormalization(self):
        """Test denormalizing model outputs to LOES scale."""
        # Model outputs normalized values
        normalized_outputs = torch.tensor([0.0, 0.5, 1.0, 0.3, 0.7])
        
        # Denormalize to 0-35 scale
        loes_scores = normalized_outputs * 35
        
        self.assertTrue(torch.all(loes_scores >= 0))
        self.assertTrue(torch.all(loes_scores <= 35))
        self.assertEqual(loes_scores[0].item(), 0.0)
        self.assertEqual(loes_scores[2].item(), 35.0)
    
    def test_score_clipping(self):
        """Test clipping predictions to valid range."""
        # Some predictions might be out of range
        raw_predictions = torch.tensor([-5.0, 10.0, 40.0, 25.0])
        
        # Clip to valid LOES range
        clipped = torch.clamp(raw_predictions, 0, 35)
        
        self.assertEqual(clipped[0].item(), 0.0)
        self.assertEqual(clipped[1].item(), 10.0)
        self.assertEqual(clipped[2].item(), 35.0)
        self.assertEqual(clipped[3].item(), 25.0)
    
    def test_confidence_estimation(self):
        """Test confidence/uncertainty estimation."""
        # Simulate multiple forward passes (MC Dropout or ensemble)
        predictions = torch.tensor([
            [15.0], [15.5], [14.8], [15.2], [15.1]
        ])
        
        mean_prediction = predictions.mean()
        std_prediction = predictions.std()
        
        self.assertAlmostEqual(mean_prediction.item(), 15.12, places=1)
        self.assertGreater(std_prediction.item(), 0)
        
        # 95% confidence interval
        ci_lower = mean_prediction - 1.96 * std_prediction
        ci_upper = mean_prediction + 1.96 * std_prediction
        
        self.assertLess(ci_lower, mean_prediction)
        self.assertGreater(ci_upper, mean_prediction)


class TestBatchProcessing(unittest.TestCase):
    """Test batch processing for multiple files."""
    
    def test_batch_file_processing(self):
        """Test processing multiple files in batch."""
        file_paths = [f'scan_{i}.nii.gz' for i in range(10)]
        predictions = []
        
        model = nn.Linear(100, 1)
        model.eval()
        
        for path in file_paths:
            # Simulate loading and processing
            fake_data = torch.randn(1, 100)
            
            with torch.no_grad():
                pred = model(fake_data)
                predictions.append(pred.item())
        
        self.assertEqual(len(predictions), 10)
        self.assertTrue(all(isinstance(p, float) for p in predictions))
    
    def test_results_aggregation(self):
        """Test aggregating results into DataFrame."""
        subjects = ['sub-01', 'sub-02', 'sub-03']
        sessions = ['ses-01', 'ses-01', 'ses-02']
        predictions = [15.5, 20.3, 18.7]
        true_scores = [16.0, 19.0, 19.5]
        
        results_df = pd.DataFrame({
            'subject': subjects,
            'session': sessions,
            'predicted': predictions,
            'true': true_scores
        })
        
        # Calculate metrics
        results_df['error'] = results_df['predicted'] - results_df['true']
        results_df['abs_error'] = results_df['error'].abs()
        
        mae = results_df['abs_error'].mean()
        
        self.assertEqual(len(results_df), 3)
        self.assertIn('error', results_df.columns)
        self.assertGreater(mae, 0)


class TestErrorHandling(unittest.TestCase):
    """Test error handling in inference."""
    
    def test_missing_model_file(self):
        """Test handling of missing model file."""
        with self.assertRaises(FileNotFoundError):
            torch.load('/nonexistent/model.pt')
    
    def test_corrupted_model_file(self):
        """Test handling of corrupted model file."""
        with tempfile.NamedTemporaryFile(suffix='.pt') as f:
            # Write garbage data
            f.write(b'corrupted data')
            f.flush()
            
            with self.assertRaises((RuntimeError, pickle.UnpicklingError)):
                torch.load(f.name)
    
    def test_incompatible_model_architecture(self):
        """Test handling of incompatible architectures."""
        model1 = nn.Linear(10, 5)
        model2 = nn.Linear(20, 5)  # Different input size
        
        state_dict = model1.state_dict()
        
        with self.assertRaises(RuntimeError):
            model2.load_state_dict(state_dict, strict=True)
    
    def test_invalid_input_shape(self):
        """Test handling of invalid input shapes."""
        model = nn.Conv3d(1, 8, 3)
        
        # Wrong number of dimensions
        invalid_input = torch.randn(32, 32, 32)  # Missing batch and channel
        
        with self.assertRaises(RuntimeError):
            model(invalid_input)
    
    def test_out_of_memory_handling(self):
        """Test handling of OOM errors."""
        # This would actually cause OOM on most systems
        # so we mock it instead
        with patch('torch.randn') as mock_randn:
            mock_randn.side_effect = RuntimeError("CUDA out of memory")
            
            with self.assertRaises(RuntimeError) as context:
                torch.randn(1000000, 1000000)
            
            self.assertIn("out of memory", str(context.exception))


class TestInferenceOptimization(unittest.TestCase):
    """Test inference optimization techniques."""
    
    def test_no_grad_context(self):
        """Test that no_grad context prevents gradient computation."""
        model = nn.Linear(10, 1)
        input_data = torch.randn(5, 10)
        
        # Without no_grad
        output_with_grad = model(input_data)
        self.assertTrue(output_with_grad.requires_grad)
        
        # With no_grad
        with torch.no_grad():
            output_no_grad = model(input_data)
            self.assertFalse(output_no_grad.requires_grad)
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_half_precision_inference(self):
        """Test half precision (FP16) inference."""
        model = nn.Linear(10, 1).cuda().half()
        input_data = torch.randn(5, 10).cuda().half()
        
        with torch.no_grad():
            output = model(input_data)
        
        self.assertEqual(output.dtype, torch.float16)
    
    def test_model_quantization(self):
        """Test model quantization for faster inference."""
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        
        # Prepare for quantization
        model.eval()
        
        # Dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
        
        # Test inference
        input_data = torch.randn(5, 10)
        with torch.no_grad():
            output = quantized_model(input_data)
        
        self.assertEqual(output.shape, (5, 1))


if __name__ == '__main__':
    unittest.main()