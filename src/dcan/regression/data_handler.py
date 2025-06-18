import logging
import os
import random
import sys

import torch
from torch.utils.data import DataLoader

from dcan.training.augmented_loes_score_dataset import AugmentedLoesScoreDataset
from dcan.regression.dsets import LoesScoreDataset


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)


class DataHandler:
    def __init__(self, input_df, output_df, use_cuda, batch_size, num_workers, config):
        """
        Initialize the DataHandler with all necessary configuration.
        
        Args:
            input_df (pd.DataFrame): Input data
            output_df (pd.DataFrame): Output data for predictions
            use_cuda (bool): Whether to use CUDA
            batch_size (int): Batch size for training
            num_workers (int): Number of worker processes
            config (Config): Configuration object containing all settings
        """
        self.input_df = input_df
        self.output_df = output_df
        self.use_cuda = use_cuda
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.config = config
        
        # Store configuration settings directly
        self.folder = config.folder
        self.augment_minority = config.use_weighted_loss
        self.augment_minority = getattr(config, 'augment_minority', False)
        if self.augment_minority:
            self.num_augmentations = getattr(config, 'num_augmentations', 3)
        
        # Create the train/validation split
        self.train_subjects, self.val_subjects = self._create_stratified_split()

    def _setup_dataset_split(self):
        """
        Create the train/validation split based on configuration.
        This is now handled entirely within the DataHandler.
        """
        if hasattr(self.config, 'use_train_validation_cols') and self.config.use_train_validation_cols:
            # Use existing training/validation flags from the DataFrame
            training_rows = self.input_df[self.input_df['training'] == 1]
            self.train_subjects = list(set(training_rows['anonymized_subject_id'].tolist()))
            
            validation_rows = self.input_df[self.input_df['validation'] == 1]
            self.val_subjects = list(set(validation_rows['anonymized_subject_id'].tolist()))
        else:
            # Split based on config split ratio
            self.train_subjects, self.val_subjects = self._create_stratified_split()
        
        # Log the split information
        logging.info(f"Train set: {len(self.train_subjects)} subjects")
        logging.info(f"Val set: {len(self.val_subjects)} subjects")


    def _create_stratified_split(self):
        """
        Create a stratified split that ensures both train and validation sets 
        have similar score distributions, including zero scores.
        """
        from sklearn.model_selection import train_test_split
        import numpy as np
        
        # Get unique subjects and their score characteristics
        subject_stats = []
        for subject in self.input_df['anonymized_subject_id'].unique():
            subject_data = self.input_df[self.input_df['anonymized_subject_id'] == subject]
            
            # Get subject's score characteristics
            scores = subject_data['loes-score'].values
            has_zero = (scores == 0).any()
            max_score = scores.max()
            mean_score = scores.mean()
            
            subject_stats.append({
                'subject': subject,
                'has_zero': has_zero,
                'max_score': max_score,
                'mean_score': mean_score,
                'score_category': 'zero' if has_zero else ('low' if max_score <= 3 else ('medium' if max_score <= 8 else 'high'))
            })
        
        # Convert to DataFrame for easier manipulation
        import pandas as pd
        subject_df = pd.DataFrame(subject_stats)
        
        log.info("=== SUBJECT DISTRIBUTION ANALYSIS ===")
        category_counts = subject_df['score_category'].value_counts()
        log.info(f"Subject categories: {dict(category_counts)}")
        
        # Stratified split by score category to ensure balanced distribution
        try:
            train_subjects, val_subjects = train_test_split(
                subject_df['subject'].values,
                test_size=1 - self.config.train_size,
                stratify=subject_df['score_category'].values,
                random_state=42  # For reproducibility
            )
        except ValueError as e:
            log.warning(f"Stratified split failed: {e}")
            log.warning("Falling back to score-based split...")
            
            # Fallback: split by mean score
            train_subjects, val_subjects = train_test_split(
                subject_df['subject'].values,
                test_size=1 - self.config.train_size,
                random_state=42
            )
        
        # Verify the split maintains distribution
        train_subjects_list = train_subjects.tolist()
        val_subjects_list = val_subjects.tolist()
        
        # Check distribution in splits
        train_df = self.input_df[self.input_df['anonymized_subject_id'].isin(train_subjects_list)]
        val_df = self.input_df[self.input_df['anonymized_subject_id'].isin(val_subjects_list)]
        
        train_zeros = (train_df['loes-score'] == 0).sum()
        val_zeros = (val_df['loes-score'] == 0).sum()
        
        log.info("=== NEW SPLIT VERIFICATION ===")
        log.info(f"Train set: {train_zeros}/{len(train_df)} ({train_zeros/len(train_df)*100:.1f}%) zero scores")
        log.info(f"Val set: {val_zeros}/{len(val_df)} ({val_zeros/len(val_df)*100:.1f}%) zero scores")
        log.info(f"Train scores: mean={train_df['loes-score'].mean():.2f}, std={train_df['loes-score'].std():.2f}")
        log.info(f"Val scores: mean={val_df['loes-score'].mean():.2f}, std={val_df['loes-score'].std():.2f}")
        
        return train_subjects_list, val_subjects_list


    def setup_cross_validation_split(self, fold_idx, k_folds):
        """
        Set up dataset split for a specific cross-validation fold.
        
        Args:
            fold_idx (int): Current fold index
            k_folds (int): Total number of folds
        """
        # Get all unique subjects
        all_subjects = list(set(self.input_df['anonymized_subject_id'].tolist()))
        
        # Stratify subjects based on whether they have ALD
        subjects_with_ald = []
        subjects_without_ald = []
        
        for subject in all_subjects:
            subject_rows = self.input_df[self.input_df['anonymized_subject_id'] == subject]
            max_loes = subject_rows['loes-score'].max()
            if max_loes > self.threshold:
                subjects_with_ald.append(subject)
            else:
                subjects_without_ald.append(subject)
        
        # Create folds
        ald_folds = self._create_folds(subjects_with_ald, k_folds)
        no_ald_folds = self._create_folds(subjects_without_ald, k_folds)
        
        # Set current fold as validation, rest as training
        val_subjects_ald = ald_folds[fold_idx]
        val_subjects_no_ald = no_ald_folds[fold_idx]
        
        train_subjects_ald = [s for i, fold_subjects in enumerate(ald_folds) for s in fold_subjects if i != fold_idx]
        train_subjects_no_ald = [s for i, fold_subjects in enumerate(no_ald_folds) for s in fold_subjects if i != fold_idx]
        
        self.train_subjects = train_subjects_ald + train_subjects_no_ald
        self.val_subjects = val_subjects_ald + val_subjects_no_ald
        
        logging.info(f"Fold {fold_idx+1} - Train set: {len(self.train_subjects)} subjects "
                    f"({len(train_subjects_ald)} with ALD, {len(train_subjects_no_ald)} without ALD)")
        logging.info(f"Fold {fold_idx+1} - Val set: {len(self.val_subjects)} subjects "
                    f"({len(val_subjects_ald)} with ALD, {len(val_subjects_no_ald)} without ALD)")
    
    def _create_folds(self, subjects, k):
        """
        Split list of subjects into k approximately equal folds.
        
        Args:
            subjects (list): List of subject IDs
            k (int): Number of folds
            
        Returns:
            list: List of k lists, each containing subjects for one fold
        """
        folds = [[] for _ in range(k)]
        for i, subject in enumerate(subjects):
            folds[i % k].append(subject)
        return folds

    def get_train_dataloader(self):
        """Returns training dataloader"""
        # Use your existing LoesScoreDataset constructor with correct parameters
        train_dataset = LoesScoreDataset(
            folder=self.folder,              # folder path where MRI files are stored
            subjects=self.train_subjects,    # list of training subject IDs
            df=self.input_df,               # input dataframe
            output_df=self.output_df,       # output dataframe
            is_val_set_bool=False,          # this is training data (enables augmentation)
            sortby_str='random'             # randomize order
        )
    
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,      # Always shuffle training data
            num_workers=self.num_workers,
            pin_memory=self.use_cuda,
            drop_last=True     # Drop incomplete batches for consistent training
        )

    def get_val_dataloader(self):
        """Returns validation dataloader"""
        # Use your existing LoesScoreDataset constructor with correct parameters
        val_dataset = LoesScoreDataset(
            folder=self.folder,              # folder path where MRI files are stored
            subjects=self.val_subjects,      # list of validation subject IDs
            df=self.input_df,               # input dataframe
            output_df=self.output_df,       # output dataframe
            is_val_set_bool=True,           # this is validation data (no augmentation)
            sortby_str='loes_score'         # sort by loes score for consistent validation
        )
        
        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,     # Don't shuffle validation data
            num_workers=self.num_workers,
            pin_memory=self.use_cuda
        )

    def _create_dataloader(self, subjects, is_val_set=False):
        """
        Create a DataLoader for the given subjects.
        
        Args:
            subjects (list): List of subject IDs
            is_val_set (bool): Whether this is a validation set
            
        Returns:
            DataLoader: Data loader for the specified subjects
        """
        if not self.folder or not os.path.exists(self.folder):
            raise ValueError(f"Invalid MRI folder path: {self.folder}")
            
        if not subjects:
            raise ValueError("No subjects provided for dataset creation")
            
        # Check how many subjects we actually have MRI files for
        valid_subjects = 0
        for subject in subjects:
            subject_rows = self.input_df[self.input_df['anonymized_subject_id'] == subject]
            if not subject_rows.empty:
                valid_subjects += 1
                
        if valid_subjects == 0:
            raise ValueError("None of the provided subjects exist in the dataset")
        elif valid_subjects < len(subjects):
            logging.warning(f"Only {valid_subjects} out of {len(subjects)} subjects exist in the dataset")
        
        # Use augmented dataset for training
        try:
            if not is_val_set and self.augment_minority:
                dataset = AugmentedLoesScoreDataset(
                    self.folder, subjects, self.input_df, self.output_df, 
                    is_val_set_bool=is_val_set,
                    augment_minority=True, 
                    num_augmentations=self.num_augmentations
                )
            else:
                dataset = LoesScoreDataset(self.folder, subjects, self.input_df, self.output_df, is_val_set_bool=is_val_set)
        except Exception as e:
            logging.error(f"Error creating dataset: {e}")
            raise
        
        batch_size = self.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        return DataLoader(dataset, batch_size=batch_size, num_workers=self.num_workers, pin_memory=self.use_cuda)