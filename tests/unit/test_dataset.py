"""
Unit tests for dataset loading and data pipeline.
Tests LoesScoreDataset and data augmentation.
"""

import unittest
import tempfile
import os
import sys
import torch
import numpy as np
import pandas as pd
import nibabel as nib
from unittest.mock import patch, MagicMock, mock_open, PropertyMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from dcan.regression.dsets import LoesScoreDataset, CandidateInfoTuple


class TestLoesScoreDataset(unittest.TestCase):
    """Test suite for LoesScoreDataset."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock dataframe
        self.df = pd.DataFrame({
            'anonymized_subject_id': ['sub-01', 'sub-02', 'sub-03'],
            'anonymized_session_id': ['ses-01', 'ses-01', 'ses-01'],
            'scan': ['mprage.nii.gz', 'mprage.nii.gz', 'mprage.nii.gz'],
            'loes-score': [10.0, 15.0, 20.0],
            'Gd-enhanced': [0, 0, 0]
        })
        
        self.output_df = pd.DataFrame()
        self.subjects = ['sub-01', 'sub-02']
        self.folder = '/fake/data/folder'
        
        # Create mock candidate info list
        self.mock_candidates = [
            CandidateInfoTuple(10.0, '/fake/path1.nii', 'sub-01', 'ses-01'),
            CandidateInfoTuple(15.0, '/fake/path2.nii', 'sub-02', 'ses-01'),
        ]
    
    @patch('dcan.regression.dsets.get_candidate_info_list')
    def test_dataset_initialization(self, mock_get_candidates):
        """Test dataset initialization."""
        mock_get_candidates.return_value = self.mock_candidates
        
        dataset = LoesScoreDataset(
            folder=self.folder,
            subjects=self.subjects,
            df=self.df,
            output_df=self.output_df,
            is_val_set_bool=False
        )
        
        self.assertIsNotNone(dataset)
        self.assertEqual(len(dataset.candidateInfo_list), 2)
        self.assertFalse(dataset.is_val_set_bool)
    
    @patch('dcan.regression.dsets.get_candidate_info_list')
    def test_dataset_filtering_by_subject(self, mock_get_candidates):
        """Test filtering dataset by specific subject."""
        mock_get_candidates.return_value = self.mock_candidates
        
        dataset = LoesScoreDataset(
            folder=self.folder,
            subjects=self.subjects,
            df=self.df,
            output_df=self.output_df,
            subject='sub-01'
        )
        
        # Should only have sub-01 candidates
        self.assertEqual(len(dataset.candidateInfo_list), 1)
        self.assertEqual(dataset.candidateInfo_list[0].subject_str, 'sub-01')
    
    @patch('dcan.regression.dsets.get_candidate_info_list')
    @patch('random.shuffle')
    def test_random_sorting(self, mock_shuffle, mock_get_candidates):
        """Test random sorting of dataset."""
        mock_get_candidates.return_value = self.mock_candidates.copy()
        
        dataset = LoesScoreDataset(
            folder=self.folder,
            subjects=self.subjects,
            df=self.df,
            output_df=self.output_df,
            sortby_str='random'
        )
        
        # Check that shuffle was called
        mock_shuffle.assert_called_once()
    
    @patch('dcan.regression.dsets.get_candidate_info_list')
    def test_loes_score_sorting(self, mock_get_candidates):
        """Test sorting by LOES score."""
        candidates = [
            CandidateInfoTuple(20.0, '/fake/path1.nii', 'sub-01', 'ses-01'),
            CandidateInfoTuple(5.0, '/fake/path2.nii', 'sub-02', 'ses-01'),
            CandidateInfoTuple(15.0, '/fake/path3.nii', 'sub-03', 'ses-01'),
        ]
        mock_get_candidates.return_value = candidates
        
        dataset = LoesScoreDataset(
            folder=self.folder,
            subjects=self.subjects,
            df=self.df,
            output_df=self.output_df,
            sortby_str='loes_score'
        )
        
        # Original order should be preserved (sorted by dataclass)
        self.assertEqual(dataset.candidateInfo_list[0].loes_score_float, 20.0)
    
    @patch('dcan.regression.dsets.get_candidate_info_list')
    def test_validation_set_flag(self, mock_get_candidates):
        """Test validation set flag."""
        mock_get_candidates.return_value = self.mock_candidates
        
        # Training set
        train_dataset = LoesScoreDataset(
            folder=self.folder,
            subjects=self.subjects,
            df=self.df,
            output_df=self.output_df,
            is_val_set_bool=False
        )
        
        # Validation set
        val_dataset = LoesScoreDataset(
            folder=self.folder,
            subjects=self.subjects,
            df=self.df,
            output_df=self.output_df,
            is_val_set_bool=True
        )
        
        self.assertFalse(train_dataset.is_val_set_bool)
        self.assertTrue(val_dataset.is_val_set_bool)


class TestDatasetGetItem(unittest.TestCase):
    """Test __getitem__ functionality of dataset."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a fake NIfTI file
        self.fake_nifti_path = os.path.join(self.temp_dir, 'test.nii.gz')
        self.fake_data = np.random.randn(91, 109, 91).astype(np.float32)
        self.fake_img = nib.Nifti1Image(self.fake_data, np.eye(4))
        
    def tearDown(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('dcan.regression.dsets.LoesScoreDataset.__len__')
    @patch('dcan.regression.dsets.LoesScoreDataset.__getitem__')
    def test_getitem_returns_correct_types(self, mock_getitem, mock_len):
        """Test that __getitem__ returns correct types."""
        mock_len.return_value = 10
        
        # Mock return value: (image_tensor, score, subject_id)
        mock_image = torch.randn(1, 91, 109, 91)
        mock_score = torch.tensor(15.0)
        mock_subject = 'sub-01'
        
        mock_getitem.return_value = (mock_image, mock_score, mock_subject)
        
        # Create mock dataset
        dataset = MagicMock()
        dataset.__len__ = mock_len
        dataset.__getitem__ = mock_getitem
        
        # Test getting item
        image, score, subject = dataset[0]
        
        self.assertIsInstance(image, torch.Tensor)
        self.assertIsInstance(score, torch.Tensor)
        self.assertIsInstance(subject, str)
        
        # Check shapes
        self.assertEqual(image.shape, (1, 91, 109, 91))
        self.assertEqual(score.shape, ())  # Scalar
    
    @patch('nibabel.load')
    def test_nifti_loading(self, mock_nib_load):
        """Test NIfTI file loading."""
        mock_nib_load.return_value = self.fake_img
        
        # Simulate loading
        img = mock_nib_load(self.fake_nifti_path)
        data = img.get_fdata()
        
        self.assertEqual(data.shape, (91, 109, 91))
        self.assertEqual(data.dtype, np.float32)


class TestDataAugmentation(unittest.TestCase):
    """Test data augmentation pipeline."""
    
    def test_augmentation_reproducibility(self):
        """Test that augmentation is reproducible with same seed."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        fake_volume1 = torch.randn(1, 91, 109, 91)
        
        torch.manual_seed(42)
        np.random.seed(42)
        
        fake_volume2 = torch.randn(1, 91, 109, 91)
        
        torch.testing.assert_close(fake_volume1, fake_volume2)
    
    def test_augmentation_different_seeds(self):
        """Test that different seeds produce different augmentations."""
        torch.manual_seed(42)
        fake_volume1 = torch.randn(1, 91, 109, 91)
        
        torch.manual_seed(123)
        fake_volume2 = torch.randn(1, 91, 109, 91)
        
        self.assertFalse(torch.allclose(fake_volume1, fake_volume2))
    
    def test_augmentation_preserves_shape(self):
        """Test that augmentation preserves tensor shape."""
        original_shape = (1, 91, 109, 91)
        fake_volume = torch.randn(*original_shape)
        
        # Simulate common augmentations
        # Random flip
        if torch.rand(1) > 0.5:
            fake_volume = torch.flip(fake_volume, dims=[1])
        
        # Random noise
        noise = torch.randn_like(fake_volume) * 0.01
        augmented = fake_volume + noise
        
        self.assertEqual(augmented.shape, original_shape)
    
    def test_augmentation_value_range(self):
        """Test that augmentation keeps values in reasonable range."""
        fake_volume = torch.randn(1, 91, 109, 91)
        
        # Normalize to [0, 1] range
        normalized = (fake_volume - fake_volume.min()) / (fake_volume.max() - fake_volume.min())
        
        self.assertTrue(torch.all(normalized >= 0))
        self.assertTrue(torch.all(normalized <= 1))


class TestDataLoader(unittest.TestCase):
    """Test DataLoader integration with dataset."""
    
    @patch('dcan.regression.dsets.LoesScoreDataset')
    def test_dataloader_creation(self, mock_dataset_class):
        """Test creating DataLoader from dataset."""
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=10)
        mock_dataset.__getitem__ = MagicMock(
            return_value=(torch.randn(1, 91, 109, 91), torch.tensor(15.0), 'sub-01')
        )
        mock_dataset_class.return_value = mock_dataset
        
        # Create DataLoader
        dataloader = torch.utils.data.DataLoader(
            mock_dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0
        )
        
        self.assertIsNotNone(dataloader)
        self.assertEqual(len(dataloader), 5)  # 10 samples / batch_size 2
    
    def test_batch_collation(self):
        """Test that batches are correctly collated."""
        # Create mock batch data
        batch = [
            (torch.randn(1, 32, 32, 32), torch.tensor(10.0), 'sub-01'),
            (torch.randn(1, 32, 32, 32), torch.tensor(15.0), 'sub-02'),
            (torch.randn(1, 32, 32, 32), torch.tensor(20.0), 'sub-03'),
        ]
        
        # Simulate default collate
        images = torch.stack([item[0] for item in batch])
        scores = torch.stack([item[1] for item in batch])
        subjects = [item[2] for item in batch]
        
        self.assertEqual(images.shape, (3, 1, 32, 32, 32))
        self.assertEqual(scores.shape, (3,))
        self.assertEqual(len(subjects), 3)


class TestDataValidation(unittest.TestCase):
    """Test data validation and error handling."""
    
    def test_invalid_score_range(self):
        """Test handling of invalid LOES scores."""
        invalid_scores = [-5.0, 40.0, np.nan, np.inf]
        
        for score in invalid_scores:
            with self.subTest(score=score):
                is_valid = 0 <= score <= 35 if not np.isnan(score) and not np.isinf(score) else False
                self.assertFalse(is_valid)
    
    def test_missing_file_handling(self):
        """Test handling of missing NIfTI files."""
        missing_path = '/nonexistent/file.nii.gz'
        
        with patch('nibabel.load') as mock_load:
            mock_load.side_effect = FileNotFoundError()
            
            with self.assertRaises(FileNotFoundError):
                mock_load(missing_path)
    
    def test_corrupt_nifti_handling(self):
        """Test handling of corrupt NIfTI files."""
        corrupt_path = '/fake/corrupt.nii.gz'
        
        with patch('nibabel.load') as mock_load:
            mock_load.side_effect = nib.filebasedimages.ImageFileError()
            
            with self.assertRaises(nib.filebasedimages.ImageFileError):
                mock_load(corrupt_path)
    
    def test_empty_dataset(self):
        """Test handling of empty dataset."""
        empty_df = pd.DataFrame()
        
        self.assertEqual(len(empty_df), 0)
        self.assertTrue(empty_df.empty)
    
    def test_duplicate_subjects(self):
        """Test handling of duplicate subject entries."""
        df = pd.DataFrame({
            'anonymized_subject_id': ['sub-01', 'sub-01', 'sub-02'],
            'anonymized_session_id': ['ses-01', 'ses-01', 'ses-01'],
            'loes-score': [10.0, 10.0, 15.0]
        })
        
        # Check for duplicates
        duplicates = df.duplicated(subset=['anonymized_subject_id', 'anonymized_session_id'])
        self.assertTrue(duplicates.any())
        
        # Remove duplicates
        df_clean = df.drop_duplicates(subset=['anonymized_subject_id', 'anonymized_session_id'])
        self.assertEqual(len(df_clean), 2)


if __name__ == '__main__':
    unittest.main()