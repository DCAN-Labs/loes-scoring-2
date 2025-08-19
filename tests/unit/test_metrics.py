"""
Unit tests for metrics module.
Tests statistical metrics used for model evaluation.
"""

import unittest
import numpy as np
import math
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from dcan.metrics import get_standardized_rmse


class TestMetrics(unittest.TestCase):
    """Test suite for metrics calculations."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.target_perfect = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.preds_perfect = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        self.target_noisy = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.preds_noisy = np.array([1.2, 1.8, 3.3, 3.7, 5.2])
        
        self.target_constant = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
        self.preds_constant = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
        
        self.target_outlier = np.array([1.0, 2.0, 3.0, 4.0, 35.0])
        self.preds_outlier = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    def test_perfect_prediction(self):
        """Test SRMSE for perfect predictions."""
        srmse = get_standardized_rmse(self.target_perfect, self.preds_perfect)
        self.assertAlmostEqual(srmse, 0.0, places=5)
    
    def test_noisy_prediction(self):
        """Test SRMSE for noisy predictions."""
        srmse = get_standardized_rmse(self.target_noisy, self.preds_noisy)
        # RMSE should be > 0 for imperfect predictions
        self.assertGreater(srmse, 0.0)
        # But should be reasonable for small noise
        self.assertLess(srmse, 1.0)
    
    def test_constant_prediction_edge_case(self):
        """Test SRMSE when predictions have zero variance."""
        # When all predictions are the same, std dev is 0
        # This should raise an error or return inf
        with self.assertRaises(ZeroDivisionError):
            # Predictions with zero variance
            constant_preds = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
            variable_target = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            get_standardized_rmse(variable_target, constant_preds)
    
    def test_outlier_handling(self):
        """Test SRMSE with outliers in target."""
        srmse = get_standardized_rmse(self.target_outlier, self.preds_outlier)
        # Should have high error due to outlier
        self.assertGreater(srmse, 1.0)
    
    def test_negative_values(self):
        """Test SRMSE with negative values."""
        target_neg = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        preds_neg = np.array([-1.8, -1.2, 0.1, 0.9, 2.1])
        srmse = get_standardized_rmse(target_neg, preds_neg)
        self.assertGreater(srmse, 0.0)
    
    def test_large_scale_values(self):
        """Test SRMSE with Loes score scale (0-35)."""
        target_loes = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0])
        preds_loes = np.array([1.0, 4.5, 11.0, 14.0, 21.0, 24.5, 31.0, 34.0])
        srmse = get_standardized_rmse(target_loes, preds_loes)
        self.assertGreater(srmse, 0.0)
        self.assertLess(srmse, 1.0)  # Should be normalized
    
    def test_empty_arrays(self):
        """Test SRMSE with empty arrays."""
        with self.assertRaises((ValueError, IndexError, ZeroDivisionError)):
            get_standardized_rmse(np.array([]), np.array([]))
    
    def test_mismatched_lengths(self):
        """Test SRMSE with mismatched array lengths."""
        target_short = np.array([1.0, 2.0, 3.0])
        preds_long = np.array([1.0, 2.0, 3.0, 4.0])
        with self.assertRaises(ValueError):
            get_standardized_rmse(target_short, preds_long)
    
    def test_single_element(self):
        """Test SRMSE with single element arrays."""
        with self.assertRaises((ValueError, ZeroDivisionError)):
            # Single element has no variance
            get_standardized_rmse(np.array([5.0]), np.array([5.0]))
    
    def test_numerical_stability(self):
        """Test SRMSE numerical stability with very small differences."""
        target_small = np.array([1.0, 1.0001, 1.0002, 1.0003, 1.0004])
        preds_small = np.array([1.0, 1.0001, 1.0002, 1.0003, 1.0004])
        srmse = get_standardized_rmse(target_small, preds_small)
        self.assertAlmostEqual(srmse, 0.0, places=3)


class TestMetricsIntegration(unittest.TestCase):
    """Integration tests for metrics in training context."""
    
    def test_batch_metrics_calculation(self):
        """Test metrics calculation on batched predictions."""
        batch_size = 32
        targets = np.random.uniform(0, 35, batch_size)
        preds = targets + np.random.normal(0, 2, batch_size)
        
        srmse = get_standardized_rmse(targets, preds)
        
        # Check reasonable bounds for noisy predictions
        self.assertGreater(srmse, 0.0)
        self.assertLess(srmse, 5.0)
    
    def test_epoch_aggregation(self):
        """Test aggregating metrics across multiple batches."""
        all_targets = []
        all_preds = []
        
        for _ in range(10):  # 10 batches
            batch_targets = np.random.uniform(0, 35, 16)
            batch_preds = batch_targets + np.random.normal(0, 1.5, 16)
            all_targets.extend(batch_targets)
            all_preds.extend(batch_preds)
        
        final_srmse = get_standardized_rmse(
            np.array(all_targets), 
            np.array(all_preds)
        )
        
        self.assertIsInstance(final_srmse, float)
        self.assertFalse(np.isnan(final_srmse))
        self.assertFalse(np.isinf(final_srmse))


if __name__ == '__main__':
    unittest.main()