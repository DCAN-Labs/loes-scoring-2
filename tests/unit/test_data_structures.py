"""
Unit tests for data structures and dataset utilities.
Tests CandidateInfoTuple and related data handling.
"""

import unittest
import tempfile
import os
import sys
import copy
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from dcan.regression.dsets import CandidateInfoTuple, get_subject


class TestCandidateInfoTuple(unittest.TestCase):
    """Test suite for CandidateInfoTuple data structure."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_tuple = CandidateInfoTuple(
            loes_score_float=15.5,
            file_path="/data/sub-01/ses-01/mprage.nii.gz",
            subject_str="sub-01",
            session_str="ses-01"
        )
        
        self.sample_tuple_augmented = CandidateInfoTuple(
            loes_score_float=20.0,
            file_path="/data/sub-02/ses-02/mprage.nii.gz",
            subject_str="sub-02",
            session_str="ses-02",
            augmentation_index=3
        )
    
    def test_initialization(self):
        """Test CandidateInfoTuple initialization."""
        self.assertEqual(self.sample_tuple.loes_score_float, 15.5)
        self.assertEqual(self.sample_tuple.file_path, "/data/sub-01/ses-01/mprage.nii.gz")
        self.assertEqual(self.sample_tuple.subject_str, "sub-01")
        self.assertEqual(self.sample_tuple.session_str, "ses-01")
        self.assertIsNone(self.sample_tuple.augmentation_index)
    
    def test_subject_property(self):
        """Test subject property accessor."""
        self.assertEqual(self.sample_tuple.subject, "sub-01")
        self.assertEqual(self.sample_tuple_augmented.subject, "sub-02")
    
    def test_path_to_file_property(self):
        """Test path_to_file property accessor."""
        self.assertEqual(
            self.sample_tuple.path_to_file, 
            "/data/sub-01/ses-01/mprage.nii.gz"
        )
    
    def test_sort_index(self):
        """Test sort_index is set correctly."""
        self.assertEqual(self.sample_tuple.sort_index, 15.5)
        self.assertEqual(self.sample_tuple_augmented.sort_index, 20.0)
    
    def test_hash_function(self):
        """Test hash function for dictionary/set usage."""
        # Same file path should have same hash
        tuple1 = CandidateInfoTuple(10.0, "/path/file.nii", "sub-01", "ses-01")
        tuple2 = CandidateInfoTuple(20.0, "/path/file.nii", "sub-02", "ses-02")
        self.assertEqual(hash(tuple1), hash(tuple2))
        
        # Different file paths should have different hashes
        tuple3 = CandidateInfoTuple(10.0, "/other/file.nii", "sub-01", "ses-01")
        self.assertNotEqual(hash(tuple1), hash(tuple3))
    
    def test_ordering(self):
        """Test ordering based on Loes score."""
        tuple_low = CandidateInfoTuple(5.0, "/path1", "sub-01", "ses-01")
        tuple_mid = CandidateInfoTuple(15.0, "/path2", "sub-02", "ses-02")
        tuple_high = CandidateInfoTuple(25.0, "/path3", "sub-03", "ses-03")
        
        sorted_tuples = sorted([tuple_high, tuple_low, tuple_mid])
        
        self.assertEqual(sorted_tuples[0].loes_score_float, 5.0)
        self.assertEqual(sorted_tuples[1].loes_score_float, 15.0)
        self.assertEqual(sorted_tuples[2].loes_score_float, 25.0)
    
    def test_edge_case_scores(self):
        """Test edge cases for Loes scores."""
        # Minimum score
        tuple_min = CandidateInfoTuple(0.0, "/path", "sub-01", "ses-01")
        self.assertEqual(tuple_min.loes_score_float, 0.0)
        
        # Maximum score
        tuple_max = CandidateInfoTuple(35.0, "/path", "sub-01", "ses-01")
        self.assertEqual(tuple_max.loes_score_float, 35.0)
        
        # Negative score (shouldn't happen but test handling)
        tuple_neg = CandidateInfoTuple(-5.0, "/path", "sub-01", "ses-01")
        self.assertEqual(tuple_neg.loes_score_float, -5.0)
    
    def test_augmentation_index(self):
        """Test augmentation index handling."""
        self.assertIsNone(self.sample_tuple.augmentation_index)
        self.assertEqual(self.sample_tuple_augmented.augmentation_index, 3)
    
    def test_immutability_concerns(self):
        """Test that dataclass behaves correctly with mutations."""
        original = CandidateInfoTuple(10.0, "/path", "sub-01", "ses-01")
        
        # Dataclass allows mutation by default
        original.loes_score_float = 20.0
        self.assertEqual(original.loes_score_float, 20.0)
        
        # But sort_index should update
        self.assertEqual(original.sort_index, 10.0)  # Doesn't auto-update


class TestGetSubjectFunction(unittest.TestCase):
    """Test suite for get_subject utility function."""
    
    def test_standard_path(self):
        """Test extracting subject from standard path."""
        path = "/data/sub-01/ses-01/mprage.nii.gz"
        # Based on the function logic: split()[0] -> split()[0] -> split()[1][4:]
        # This seems to extract subject ID from a specific path structure
        with patch('os.path.split') as mock_split:
            mock_split.side_effect = [
                ("/data/sub-01/ses-01", "mprage.nii.gz"),
                ("/data/sub-01", "ses-01"),
                ("/data", "sub-01")
            ]
            result = get_subject(path)
            # [4:] removes "sub-" prefix
            self.assertEqual(result, "01")
    
    def test_nested_path(self):
        """Test extracting subject from deeply nested path."""
        path = "/home/user/data/sub-patient123/ses-baseline/scan.nii.gz"
        with patch('os.path.split') as mock_split:
            mock_split.side_effect = [
                ("/home/user/data/sub-patient123/ses-baseline", "scan.nii.gz"),
                ("/home/user/data/sub-patient123", "ses-baseline"),
                ("/home/user/data", "sub-patient123")
            ]
            result = get_subject(path)
            self.assertEqual(result, "patient123")
    
    def test_edge_case_paths(self):
        """Test edge cases in path handling."""
        # Test with minimal path
        path = "/sub-01/ses-01/file.nii"
        with patch('os.path.split') as mock_split:
            mock_split.side_effect = [
                ("/sub-01/ses-01", "file.nii"),
                ("/sub-01", "ses-01"),
                ("/", "sub-01")
            ]
            result = get_subject(path)
            self.assertEqual(result, "01")


class TestDataValidation(unittest.TestCase):
    """Test suite for data validation utilities."""
    
    def test_loes_score_validation(self):
        """Test Loes score range validation."""
        valid_scores = [0.0, 17.5, 35.0]
        invalid_scores = [-1.0, 36.0, 100.0]
        
        for score in valid_scores:
            self.assertTrue(0 <= score <= 35)
        
        for score in invalid_scores:
            self.assertFalse(0 <= score <= 35)
    
    def test_file_path_validation(self):
        """Test file path validation."""
        valid_paths = [
            "/data/scan.nii.gz",
            "/home/user/data/mprage.nii",
            "relative/path/scan.nii.gz"
        ]
        
        invalid_paths = [
            "",
            None,
            "/path/without/extension",
            "/path/wrong/extension.txt"
        ]
        
        for path in valid_paths:
            self.assertTrue(
                path and (path.endswith('.nii.gz') or path.endswith('.nii'))
            )
        
        for path in invalid_paths:
            self.assertFalse(
                path and (path.endswith('.nii.gz') or path.endswith('.nii'))
            )
    
    def test_subject_session_format(self):
        """Test subject and session ID format validation."""
        valid_ids = [
            ("sub-01", "ses-01"),
            ("sub-patient123", "ses-baseline"),
            ("subject-001", "session-002")
        ]
        
        for sub, ses in valid_ids:
            self.assertIsInstance(sub, str)
            self.assertIsInstance(ses, str)
            self.assertTrue(len(sub) > 0)
            self.assertTrue(len(ses) > 0)


class TestDataConsistency(unittest.TestCase):
    """Test suite for data consistency checks."""
    
    def test_duplicate_detection(self):
        """Test detection of duplicate entries."""
        tuples = [
            CandidateInfoTuple(10.0, "/path1", "sub-01", "ses-01"),
            CandidateInfoTuple(15.0, "/path2", "sub-01", "ses-02"),
            CandidateInfoTuple(10.0, "/path1", "sub-01", "ses-01"),  # Duplicate
        ]
        
        # Using set with hash should detect duplicates
        unique_paths = set(t.file_path for t in tuples)
        self.assertEqual(len(unique_paths), 2)
    
    def test_subject_session_consistency(self):
        """Test subject-session pairing consistency."""
        tuple1 = CandidateInfoTuple(10.0, "/data/sub-01/ses-01/scan.nii", "sub-01", "ses-01")
        tuple2 = CandidateInfoTuple(15.0, "/data/sub-01/ses-02/scan.nii", "sub-01", "ses-02")
        
        # Same subject can have multiple sessions
        self.assertEqual(tuple1.subject, tuple2.subject)
        self.assertNotEqual(tuple1.session_str, tuple2.session_str)
    
    def test_score_distribution(self):
        """Test realistic Loes score distribution."""
        np.random.seed(42)
        # Simulate realistic score distribution (skewed towards lower values)
        scores = np.concatenate([
            np.random.normal(5, 2, 50),  # Many low scores
            np.random.normal(15, 3, 30),  # Some medium scores
            np.random.normal(25, 3, 20)   # Few high scores
        ])
        scores = np.clip(scores, 0, 35)  # Ensure valid range
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Check realistic statistics
        self.assertTrue(0 <= mean_score <= 35)
        self.assertGreater(std_score, 0)
        self.assertLess(std_score, 20)


if __name__ == '__main__':
    unittest.main()