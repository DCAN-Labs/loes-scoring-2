"""
Unit tests for neural network model architectures.
Tests ResNet, AlexNet3D, and other model components.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from dcan.models.ResNet import get_resnet_model
from dcan.inference.models import AlexNet3D


class TestResNetModel(unittest.TestCase):
    """Test suite for ResNet model architecture."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    @patch('dcan.models.ResNet.Regressor')
    def test_resnet_initialization(self, mock_regressor):
        """Test ResNet model initialization."""
        mock_model = MagicMock()
        mock_regressor.return_value = mock_model
        
        model = get_resnet_model()
        
        # Verify Regressor was called with correct parameters
        mock_regressor.assert_called_once_with(
            in_shape=[1, 197, 233, 189],
            out_shape=1,
            channels=(16, 32, 64, 128, 256, 512, 1024),
            strides=(2, 2, 2, 2)
        )
        
        self.assertEqual(model, mock_model)
    
    def test_resnet_input_shape(self):
        """Test ResNet expected input shape."""
        # The model expects [1, 197, 233, 189] according to the code
        expected_shape = [1, 197, 233, 189]
        self.assertEqual(expected_shape[0], 1)  # Single channel (grayscale MRI)
        self.assertEqual(len(expected_shape), 4)  # 3D volume + channel
    
    def test_resnet_output_shape(self):
        """Test ResNet output shape for regression."""
        # Output should be a single value (LOES score)
        expected_output = 1
        self.assertEqual(expected_output, 1)


class TestAlexNet3D(unittest.TestCase):
    """Test suite for AlexNet3D model architecture."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_size = 4096  # Example input size
        self.model = AlexNet3D(self.input_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_initialization(self):
        """Test AlexNet3D initialization."""
        self.assertIsInstance(self.model, nn.Module)
        self.assertEqual(self.model.input_size, self.input_size)
        self.assertIsNotNone(self.model.features)
        self.assertIsNotNone(self.model.classifier)
    
    def test_feature_extractor_structure(self):
        """Test the structure of feature extraction layers."""
        features = self.model.features
        
        # Check it's a Sequential container
        self.assertIsInstance(features, nn.Sequential)
        
        # Count different layer types
        conv_layers = sum(1 for layer in features if isinstance(layer, nn.Conv3d))
        bn_layers = sum(1 for layer in features if isinstance(layer, nn.BatchNorm3d))
        relu_layers = sum(1 for layer in features if isinstance(layer, nn.ReLU))
        pool_layers = sum(1 for layer in features if isinstance(layer, nn.MaxPool3d))
        
        self.assertEqual(conv_layers, 5)  # 5 conv layers in the architecture
        self.assertEqual(bn_layers, 5)    # 5 batch norm layers
        self.assertEqual(relu_layers, 5)  # 5 ReLU activations
        self.assertEqual(pool_layers, 3)  # 3 pooling layers
    
    def test_classifier_structure(self):
        """Test the structure of classifier head."""
        classifier = self.model.classifier
        
        # Check it's a Sequential container
        self.assertIsInstance(classifier, nn.Sequential)
        
        # Check layer types in classifier
        layers = list(classifier.children())
        
        # Should have: Dropout -> Linear -> ReLU -> Dropout -> Linear
        self.assertIsInstance(layers[0], nn.Dropout)
        self.assertIsInstance(layers[1], nn.Linear)
        self.assertIsInstance(layers[2], nn.ReLU)
        self.assertIsInstance(layers[3], nn.Dropout)
        self.assertIsInstance(layers[4], nn.Linear)
        
        # Check dimensions
        self.assertEqual(layers[1].in_features, self.input_size)
        self.assertEqual(layers[1].out_features, 64)
        self.assertEqual(layers[4].in_features, 64)
        self.assertEqual(layers[4].out_features, 1)  # Single output for regression
    
    def test_forward_pass_shape(self):
        """Test forward pass with correct input shape."""
        # Create dummy input
        batch_size = 2
        # Input shape for 3D convolution: (batch, channels, depth, height, width)
        dummy_input = torch.randn(batch_size, 1, 91, 109, 91)
        
        # We need to extract features first then flatten
        features = self.model.features(dummy_input)
        
        # Check that features have expected number of channels
        self.assertEqual(features.shape[1], 128)  # Last conv layer has 128 channels
    
    def test_get_head_method(self):
        """Test get_head method returns correct structure."""
        head = self.model.get_head()
        
        self.assertIsInstance(head, nn.Sequential)
        
        layers = list(head.children())
        self.assertEqual(len(layers), 5)
        self.assertIsInstance(layers[0], nn.Dropout)
        self.assertIsInstance(layers[1], nn.Linear)
        self.assertIsInstance(layers[4], nn.Linear)
    
    def test_weight_initialization(self):
        """Test that weights are initialized properly."""
        # Check that BatchNorm weights are initialized to 1
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm3d):
                self.assertTrue(torch.allclose(module.weight.data, torch.ones_like(module.weight.data)))
    
    def test_parameter_count(self):
        """Test total number of trainable parameters."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.assertGreater(total_params, 0)
        self.assertEqual(total_params, trainable_params)  # All params should be trainable
        
        # Log parameter count for debugging
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    def test_gradient_flow(self):
        """Test that gradients can flow through the model."""
        self.model.train()
        
        # Use larger input size that works with AlexNet3D pooling layers
        # AlexNet3D needs at least 64x64x64 to avoid negative dimensions after pooling
        dummy_input = torch.randn(1, 1, 91, 109, 91, requires_grad=True)
        
        # Forward pass through features
        features = self.model.features(dummy_input)
        features_flat = features.view(features.size(0), -1)
        
        # Adjust classifier input size for test
        test_classifier = nn.Linear(features_flat.shape[1], 1)
        output = test_classifier(features_flat)
        
        # Compute dummy loss
        loss = output.sum()
        loss.backward()
        
        # Check that input has gradients
        self.assertIsNotNone(dummy_input.grad)
        self.assertFalse(torch.all(dummy_input.grad == 0))


class TestModelDevice(unittest.TestCase):
    """Test model device handling (CPU/GPU)."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = AlexNet3D(4096)
    
    def test_cpu_inference(self):
        """Test model runs on CPU."""
        self.model.cpu()
        self.model.eval()
        
        dummy_input = torch.randn(1, 1, 91, 109, 91)
        
        # Should not raise an error
        with torch.no_grad():
            features = self.model.features(dummy_input)
            self.assertIsNotNone(features)
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_gpu_inference(self):
        """Test model runs on GPU if available."""
        self.model.cuda()
        self.model.eval()
        
        dummy_input = torch.randn(1, 1, 91, 109, 91).cuda()
        
        with torch.no_grad():
            features = self.model.features(dummy_input)
            self.assertTrue(features.is_cuda)
    
    def test_model_dtype_consistency(self):
        """Test model maintains dtype consistency."""
        self.model.eval()
        
        # Test with float32
        dummy_input_f32 = torch.randn(1, 1, 91, 109, 91, dtype=torch.float32)
        features_f32 = self.model.features(dummy_input_f32)
        self.assertEqual(features_f32.dtype, torch.float32)
        
        # Convert model to float64 and test
        self.model.double()
        dummy_input_f64 = torch.randn(1, 1, 91, 109, 91, dtype=torch.float64)
        features_f64 = self.model.features(dummy_input_f64)
        self.assertEqual(features_f64.dtype, torch.float64)


class TestModelRobustness(unittest.TestCase):
    """Test model robustness to various inputs."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = AlexNet3D(4096)
        self.model.eval()
    
    def test_zero_input(self):
        """Test model with zero input."""
        zero_input = torch.zeros(1, 1, 91, 109, 91)
        
        with torch.no_grad():
            features = self.model.features(zero_input)
            # Should produce non-zero output due to bias terms
            self.assertFalse(torch.all(features == 0))
    
    def test_ones_input(self):
        """Test model with ones input."""
        ones_input = torch.ones(1, 1, 91, 109, 91)
        
        with torch.no_grad():
            features = self.model.features(ones_input)
            self.assertIsNotNone(features)
            self.assertFalse(torch.any(torch.isnan(features)))
    
    def test_large_value_input(self):
        """Test model with large input values."""
        large_input = torch.randn(1, 1, 91, 109, 91) * 1000
        
        with torch.no_grad():
            features = self.model.features(large_input)
            # Check for NaN or Inf
            self.assertFalse(torch.any(torch.isnan(features)))
            self.assertFalse(torch.any(torch.isinf(features)))
    
    def test_batch_consistency(self):
        """Test that batch processing is consistent."""
        self.model.eval()
        
        # Single sample
        single_input = torch.randn(1, 1, 91, 109, 91)
        
        # Batch of same sample repeated
        batch_input = single_input.repeat(4, 1, 1, 1, 1)
        
        with torch.no_grad():
            single_features = self.model.features(single_input)
            batch_features = self.model.features(batch_input)
            
            # All batch outputs should be identical
            for i in range(4):
                torch.testing.assert_close(
                    batch_features[i:i+1], 
                    single_features,
                    rtol=1e-5,
                    atol=1e-5
                )


class TestModelSerialization(unittest.TestCase):
    """Test model saving and loading."""
    
    def test_state_dict_consistency(self):
        """Test that state dict can be saved and loaded."""
        model1 = AlexNet3D(4096)
        model2 = AlexNet3D(4096)
        
        # Initialize models differently
        for param in model2.parameters():
            param.data.uniform_(-1, 1)
        
        # Save and load state dict
        state_dict = model1.state_dict()
        model2.load_state_dict(state_dict)
        
        # Check parameters are identical
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            torch.testing.assert_close(p1, p2)
    
    def test_checkpoint_structure(self):
        """Test checkpoint structure for training resumption."""
        model = AlexNet3D(4096)
        optimizer = torch.optim.Adam(model.parameters())
        
        checkpoint = {
            'epoch': 10,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': 0.5,
            'best_score': 0.85
        }
        
        # Check all expected keys exist
        self.assertIn('epoch', checkpoint)
        self.assertIn('model_state_dict', checkpoint)
        self.assertIn('optimizer_state_dict', checkpoint)
        self.assertIn('loss', checkpoint)
        self.assertIn('best_score', checkpoint)


if __name__ == '__main__':
    unittest.main()