import logging
import os
import random
import sys

import torch
from torch.utils.data import DataLoader

from dcan.data_sets.dsets import LoesScoreDataset
from dcan.training.augmented_loes_score_dataset import AugmentedLoesScoreDataset


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
        self.augment_minority = config.augment_minority
        self.num_augmentations = config.num_augmentations
        self.split_ratio = config.split_ratio
        self.threshold = config.threshold
        
        # Initialize subject lists
        self.train_subjects = []
        self.val_subjects = []
        
        # Create the train/validation split
        self._setup_dataset_split()

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
        Create a stratified split of subjects based on their ALD status.
        
        Returns:
            tuple: (train_subjects, val_subjects)
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
        
        # Shuffle to randomize
        random.shuffle(subjects_with_ald)
        random.shuffle(subjects_without_ald)
        
        # Calculate split indices
        ald_split_idx = int(len(subjects_with_ald) * self.split_ratio)
        no_ald_split_idx = int(len(subjects_without_ald) * self.split_ratio)
        
        # Create train and validation lists
        train_subjects = subjects_with_ald[:ald_split_idx] + subjects_without_ald[:no_ald_split_idx]
        val_subjects = subjects_with_ald[ald_split_idx:] + subjects_without_ald[no_ald_split_idx:]
        
        return train_subjects, val_subjects
    
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
        """
        Get the training data loader.
        
        Returns:
            DataLoader: Training data loader
        """
        return self._create_dataloader(self.train_subjects, is_val_set=False)
    
    def get_val_dataloader(self):
        """
        Get the validation data loader.
        
        Returns:
            DataLoader: Validation data loader
        """
        return self._create_dataloader(self.val_subjects, is_val_set=True)
    
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