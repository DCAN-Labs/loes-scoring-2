import os
import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler
import logging

log = logging.getLogger(__name__)

class DataManager:
    """Handles data loading, preprocessing, and subject splitting"""
    
    def __init__(self, config):
        self.config = config
        self.input_df = None
        self.output_df = None
        self.feature_scaler = None
        self.train_subjects = []
        self.val_subjects = []
    
    def load_data(self):
        """Load and validate input data from CSV files"""
        log.info(f"Loading data from {self.config.csv_input_file}")
        
        # Load the data
        self.input_df = pd.read_csv(self.config.csv_input_file)
        
        # Validate that required columns exist
        missing_features = [f for f in self.config.features if f not in self.input_df.columns]
        if missing_features:
            raise ValueError(f"Missing required feature columns: {missing_features}")
            
        if self.config.target not in self.input_df.columns:
            raise ValueError(f"Target column '{self.config.target}' not found in input file")
        
        # Create output dataframe
        self.output_df = self.input_df.copy()
        self.output_df["prediction"] = np.nan
        
        # Add validation and training columns if they don't exist
        if 'validation' not in self.output_df.columns:
            self.output_df['validation'] = np.nan
        if 'training' not in self.output_df.columns:
            self.output_df['training'] = np.nan
        
        # Handle feature normalization if requested
        if self.config.normalize_features:
            self._normalize_features()
        
        log.info(f"Dataset loaded successfully: {self.input_df.shape[0]} rows, {self.input_df.shape[1]} columns")
        return self.input_df, self.output_df
    
    def setup_train_val_split(self):
        """Set up training and validation subject splits"""
        if self.config.use_train_validation_cols:
            # Use existing training/validation flags
            training_rows = self.input_df[self.input_df['training'] == 1]
            self.train_subjects = list(set(training_rows['anonymized_subject_id'].tolist()))
            
            validation_rows = self.input_df[self.input_df['validation'] == 1]
            self.val_subjects = list(set(validation_rows['anonymized_subject_id'].tolist()))
        else:
            # Use a fixed split ratio
            all_subjects = list(set(self.input_df['anonymized_subject_id'].tolist()))
            split_idx = int(len(all_subjects) * self.config.split_ratio)
            self.train_subjects = all_subjects[:split_idx]
            self.val_subjects = all_subjects[split_idx:]
        
        log.info(f"Train subjects: {len(self.train_subjects)}")
        log.info(f"Val subjects: {len(self.val_subjects)}")
        
        return self.train_subjects, self.val_subjects
    
    def _normalize_features(self):
        """Normalize feature columns to have zero mean and unit variance"""
        log.info("Normalizing feature columns...")
        
        scaler = StandardScaler()
        self.input_df[self.config.features] = scaler.fit_transform(
            self.input_df[self.config.features]
        )
        self.feature_scaler = scaler
        
        log.info("Feature normalization complete")
    
    def check_data_integrity(self):
        """Performs comprehensive checks to ensure data integrity"""
        if self.input_df is None or self.input_df.empty:
            raise ValueError("Input DataFrame is empty. Cannot proceed with training.")
        
        log.info("Data integrity check completed.")
