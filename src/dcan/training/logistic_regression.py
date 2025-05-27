import argparse
import datetime
import json
import os
import random
import sys
from typing import List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR, OneCycleLR
from sklearn.metrics import accuracy_score, auc, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import logging

# Fix matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from tqdm import tqdm

from dcan.data_sets.dsets import LoesScoreDataset
from dcan.training.augmented_loes_score_dataset import AugmentedLoesScoreDataset
from dcan.training.metrics.participant_visible_error import score
from mri_logistic_regression import get_mri_logistic_regression_model


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
    def __init__(self, df, output_df, use_cuda, batch_size, num_workers):
        self.df = df
        self.output_df = output_df
        self.use_cuda = use_cuda
        self.batch_size = batch_size
        self.num_workers = num_workers

    def init_dl(self, folder, subjects, is_val_set: bool = False):
        if not folder or not os.path.exists(folder):
            raise ValueError(f"Invalid MRI folder path: {folder}")
            
        if not subjects:
            raise ValueError("No subjects provided for dataset creation")
            
        # Check how many subjects we actually have MRI files for
        valid_subjects = 0
        for subject in subjects:
            subject_rows = self.df[self.df['anonymized_subject_id'] == subject]
            if not subject_rows.empty:
                valid_subjects += 1
                
        if valid_subjects == 0:
            raise ValueError("None of the provided subjects exist in the dataset")
        elif valid_subjects < len(subjects):
            log.warning(f"Only {valid_subjects} out of {len(subjects)} subjects exist in the dataset")
        
        # Use augmented dataset for training
        try:
            if not is_val_set and hasattr(self, 'augment_minority') and self.augment_minority:
                dataset = AugmentedLoesScoreDataset(
                    folder, subjects, self.df, self.output_df, 
                    is_val_set_bool=is_val_set,
                    augment_minority=True, 
                    num_augmentations=3
                )
            else:
                dataset = LoesScoreDataset(folder, subjects, self.df, self.output_df, is_val_set_bool=is_val_set)
        except Exception as e:
            log.error(f"Error creating dataset: {e}")
            raise
    
        batch_size = self.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        # Reduce num_workers and add persistent_workers for stability
        num_workers = min(self.num_workers, 2)  # Limit to 2 workers max
        
        # Create dataloader kwargs
        dataloader_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'num_workers': num_workers,
            'pin_memory': self.use_cuda,
            'drop_last': False,  # Don't drop incomplete batches
        }
        
        # Only add these parameters if num_workers > 0
        if num_workers > 0:
            dataloader_kwargs['persistent_workers'] = True
            dataloader_kwargs['timeout'] = 30
        
        return DataLoader(**dataloader_kwargs)


# Metrics indices
METRICS_LABEL_NDX = 0
METRICS_PRED_NDX = 1
METRICS_PROB_NDX = 2
METRICS_LOSS_NDX = 3
METRICS_SIZE = 4


class SimpleMRIModel(nn.Module):
    def __init__(self, input_dim=None, debug=False):
        """
        A simple MLP model for MRI data classification.
        
        Args:
            input_dim (int, optional): Input feature dimension. If None, will be initialized on first forward pass.
            debug (bool): Whether to enable debug logging
        """
        super(SimpleMRIModel, self).__init__()
        self.debug = debug
        self.initialized = input_dim is not None
        self.input_dim = input_dim
        
        if self.initialized:
            self._initialize(input_dim)
    
    def _initialize(self, input_size):
        """
        Initialize model layers.
        
        Args:
            input_size (int): Dimension of input features
        """
        # Create a simple feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),  # Add dropout for regularization
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.ReLU()
        )
        self.classifier = nn.Linear(32, 1)
        self.initialized = True
        
        if self.debug:
            log.debug(f"SimpleMRIModel initialized with input size: {input_size}")
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Probability predictions
        """
        if self.debug:
            log.debug(f"Input shape: {x.shape}")
        
        # Reshape input to 2D (batch_size, features) if needed
        original_shape = x.shape
        if len(original_shape) > 2:
            batch_size = original_shape[0]
            x_flat = x.view(batch_size, -1)
        else:
            x_flat = x
        
        if self.debug:
            log.debug(f"Flattened shape: {x_flat.shape}")
        
        # Initialize on first forward pass if needed
        if not self.initialized:
            self._initialize(x_flat.size(1))
            self.input_dim = x_flat.size(1)
            if self.debug:
                log.debug(f"Model automatically initialized with input size: {self.input_dim}")
        
        # Apply feature extraction
        features = self.feature_extractor(x_flat)
        
        if self.debug:
            log.debug(f"Features shape: {features.shape}")
        
        # Apply final classification
        logits = self.classifier(features)
        
        return torch.sigmoid(logits)


# Configuration class to handle CLI arguments
class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--DEBUG', action='store_true')
        # Data parameters
        self.parser.add_argument('--csv-input-file', required=True, help="CSV data file")
        self.parser.add_argument('--csv-output-file', help="CSV output file for predictions")
        self.parser.add_argument('--features', required=True, nargs='+', help="Feature column names")
        self.parser.add_argument('--target', required=True, help="Target column name")
        self.parser.add_argument('--folder', help='Folder where MRIs are stored')
        self.parser.add_argument('--use_train_validation_cols', action='store_true')
        
        # Training parameters
        self.parser.add_argument('--batch-size', default=32, type=int, help='Batch size for training')
        self.parser.add_argument('--epochs', default=100, type=int, help='Number of epochs to train')
        self.parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')
        self.parser.add_argument('--num-workers', default=4, type=int, help='Number of worker processes')
        self.parser.add_argument('--split-ratio', default=0.8, type=float, help='Train/validation split ratio')
        self.parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'], help="Optimizer type")
        self.parser.add_argument('--scheduler', default='plateau', 
                                choices=['plateau', 'step', 'cosine', 'onecycle'], help='Learning rate scheduler')
        self.parser.add_argument('--weight-decay', default=0.0001, type=float, help='L2 regularization')
        self.parser.add_argument(
            '--model-type', 
            default='conv', 
            choices=['conv', 'simple', 'resnet3d', 'dense3d', 'efficientnet3d'],
            help='Type of model to use'
        )
        self.parser.add_argument('--augment-minority', action='store_true', 
                        help='Apply data augmentation to minority class')
        self.parser.add_argument('--num-augmentations', type=int, default=3,
                                help='Number of augmentations per minority class sample')
        
        # Output and tracking
        self.parser.add_argument('--tb-prefix', default='logistic_regression', help="Tensorboard data prefix")
        self.parser.add_argument('--model-save-location', help='Location to save model')
        self.parser.add_argument('--plot-location', help='Location to save plots')
        self.parser.add_argument('--comment', default='', help="Comment for Tensorboard run")
        self.parser.add_argument('--normalize-features', action='store_true', help='Normalize input features')
        
        # Evaluation
        self.parser.add_argument('--threshold', default=0.5, type=float, help='Classification threshold')

    def validate_args(self, args):
        """Validate command line arguments and provide warnings/errors for invalid configurations."""
        # File path validations
        if not os.path.exists(args.csv_input_file):
            raise FileNotFoundError(f"Input CSV file not found: {args.csv_input_file}")
            
        if args.folder and not os.path.exists(args.folder):
            raise FileNotFoundError(f"MRI folder not found: {args.folder}")
        
        # Numeric parameter validations
        if args.batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {args.batch_size}")
            
        if args.epochs <= 0:
            raise ValueError(f"Number of epochs must be positive, got {args.epochs}")
            
        if args.lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {args.lr}")
            
        if args.num_workers < 0:
            raise ValueError(f"Number of workers must be non-negative, got {args.num_workers}")
            
        if not (0 < args.split_ratio < 1):
            raise ValueError(f"Split ratio must be between 0 and 1, got {args.split_ratio}")
            
        if args.weight_decay < 0:
            raise ValueError(f"Weight decay must be non-negative, got {args.weight_decay}")
            
        if not (0 <= args.threshold <= 1):
            raise ValueError(f"Classification threshold must be between 0 and 1, got {args.threshold}")
    
        # Feature validation
        if not args.features:
            raise ValueError("No features specified")
            
        # Model compatibility checks
        if args.model_type in ['resnet3d', 'dense3d', 'efficientnet3d'] and not args.folder:
            raise ValueError(f"Model type '{args.model_type}' requires MRI data folder (--folder)")
        
        # Optimizer/scheduler compatibility
        if args.scheduler == 'onecycle' and args.optimizer.lower() != 'adam':
            log.warning(f"OneCycle scheduler works best with Adam optimizer, but {args.optimizer} was specified")
        
        # Augmentation parameter checks
        if args.augment_minority and args.num_augmentations <= 0:
            raise ValueError(f"Number of augmentations must be positive when augment_minority is enabled")
            
        # Warning for potentially problematic configurations
        if args.num_workers > 8:
            log.warning(f"High number of workers ({args.num_workers}) may cause memory issues")
            
        if args.batch_size > 128:
            log.warning(f"Large batch size ({args.batch_size}) may cause memory issues")
            
        return args

    def parse_args(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]
        args = self.parser.parse_args(sys_argv)
        
        # Set default model save location if not provided
        if not args.model_save_location:
            args.model_save_location = f'./logistic_model-{datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")}.pt'
        
        # Set default plot location if not provided
        if not args.plot_location and args.csv_output_file:
            args.plot_location = os.path.splitext(args.csv_output_file)[0] + "_plot.png"
        
        # Validate arguments
        args = self.validate_args(args)
        
        return args


# Training and validation loop handler
class LogisticRegressionTrainer:
    def __init__(self, model, optimizer, device, threshold=0.5):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.total_samples = 0
        self.threshold = threshold
        
        # Set up loss function
        self.loss_fn = nn.BCELoss(reduction='none')

    def find_optimal_threshold_for_pauc(self, val_dl, max_fpr=0.1):
        """
        Find the optimal threshold that maximizes pAUC (partial AUC) for the validation set.
        
        Args:
            val_dl: Validation dataloader
            max_fpr: Maximum false positive rate for pAUC calculation
        
        Returns:
            tuple: (optimal_threshold, best_pauc, metrics_dict)
        """
        from sklearn.metrics import roc_curve, auc
        
        self.model.eval()
        all_probs = []
        all_labels = []
        
        # Collect all predictions and labels
        with torch.no_grad():
            for batch_tup in val_dl:
                if len(batch_tup) == 4:
                    input_t, label_t, _, _ = batch_tup
                    label = [1.0 if l_t.item() > self.threshold else 0.0 for l_t in label_t]
                    all_labels.extend(label)
                elif len(batch_tup) == 5:
                    input_t, _, has_ald_t, _, _ = batch_tup
                    all_labels.extend(has_ald_t.float().cpu().numpy())
                
                input_g = input_t.to(self.device)
                probs = self.model(input_g).squeeze().cpu().numpy()
                
                # Handle single sample case and 0-d arrays
                if np.ndim(probs) == 0:  # Check for 0-dimensional array
                    probs = np.array([probs])
                elif np.ndim(probs) == 1:  # Already 1-dimensional, good to go
                    pass
                else:  # Multi-dimensional, flatten to 1-D
                    probs = probs.flatten()
                
                all_probs.extend(probs)
            
            # Convert to numpy arrays
            all_probs = np.array(all_probs)
            all_labels = np.array(all_labels)
            
            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
            
            # Find indices where FPR <= max_fpr
            valid_indices = np.where(fpr <= max_fpr)[0]
        
            if len(valid_indices) <= 1:
                log.warning(f"Not enough points to calculate pAUC at max_fpr={max_fpr}")
                return 0.5, 0.0, {}
            
            # Calculate pAUC for each threshold
            best_pauc = 0
            best_threshold = 0.5
            best_metrics = {}
            
            for i in valid_indices:
                if i < len(thresholds):
                    threshold = thresholds[i]
                
                    # Calculate pAUC up to current threshold
                    current_fpr = fpr[:i+1]
                    current_tpr = tpr[:i+1]
                    
                    if len(current_fpr) > 1:
                        current_pauc = auc(current_fpr, current_tpr) / max_fpr
                        
                        if current_pauc > best_pauc:
                            best_pauc = current_pauc
                            best_threshold = threshold
                            
                            # Calculate comprehensive metrics at this threshold
                            y_pred = (all_probs >= threshold).astype(int)
                            best_metrics = self._calculate_metrics(all_labels, y_pred, all_probs)
                            best_metrics['threshold'] = threshold
                            best_metrics['pauc'] = current_pauc
            
            return best_threshold, best_pauc, best_metrics

    def _calculate_metrics(self, y_true, y_pred, y_prob):
        """Calculate comprehensive classification metrics"""
        from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                    f1_score, roc_auc_score, confusion_matrix)
        
        metrics = {}
        try:
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
            metrics['sensitivity'] = metrics['recall']
            
            # Calculate AUC if we have both classes
            if len(np.unique(y_true)) > 1:
                metrics['auc'] = roc_auc_score(y_true, y_prob)
            
            # Calculate specificity
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                if (tn + fp) > 0:
                    metrics['specificity'] = tn / (tn + fp)
                else:
                    metrics['specificity'] = 0.0
                
                # Calculate PPV and NPV
                if (tp + fp) > 0:
                    metrics['ppv'] = tp / (tp + fp)
                else:
                    metrics['ppv'] = 0.0
                
                if (tn + fn) > 0:
                    metrics['npv'] = tn / (tn + fn)
                else:
                    metrics['npv'] = 0.0
            
        except Exception as e:
            log.error(f"Error calculating metrics: {e}")
        
        return metrics
    
    def train_epoch(self, epoch, train_dl):
        self.model.train()
        trn_metrics_g = torch.zeros(METRICS_SIZE, len(train_dl.dataset), device=self.device)
        
        with tqdm(total=len(train_dl), desc=f"Epoch {epoch} Training") as pbar:
            for batch_ndx, batch_tup in enumerate(train_dl):
                self.optimizer.zero_grad()
                loss_var = self._compute_batch_loss(batch_ndx, batch_tup, train_dl.batch_size, trn_metrics_g)
                loss_var.backward()
                self.optimizer.step()
                pbar.update(1)
                pbar.set_postfix({'loss': f"{loss_var.item():.4f}"})
        
        self.total_samples += len(train_dl.dataset)
        return trn_metrics_g.to('cpu')
    
    def validate_epoch(self, epoch, val_dl):
        with torch.no_grad():
            self.model.eval()
            val_metrics_g = torch.zeros(METRICS_SIZE, len(val_dl.dataset), device=self.device)
            
            with tqdm(total=len(val_dl), desc=f"Epoch {epoch} Validation") as pbar:
                for batch_ndx, batch_tup in enumerate(val_dl):
                    loss_var = self._compute_batch_loss(batch_ndx, batch_tup, val_dl.batch_size, val_metrics_g)
                    pbar.update(1)
                    pbar.set_postfix({'val_loss': f"{loss_var.item():.4f}"})
        
        return val_metrics_g.to('cpu')
    
    def _compute_batch_loss(self, batch_ndx, batch_tup, batch_size, metrics_g):
        log.debug(f'batch_tup: {batch_tup}')
        log.debug(f'type(batch_tup): {type(batch_tup)}')
        
        # Check the structure of batch_tup
        if len(batch_tup) == 4:
            # Format: (input_t, label_t, subject_str, session_str)
            input_t, label_t, _, _ = batch_tup
            # Ensure it's a float
            label_t = label_t.float()
        elif len(batch_tup) == 5:
            # Format: (input_t, _, has_ald_t, subject_str, session_str)
            input_t, _, has_ald_t, _, _ = batch_tup
            # Use has_ald_t as our label
            label_t = has_ald_t.float()
        else:
            # Unknown format, try to handle gracefully
            log.error(f"Unexpected batch tuple format: {[type(x) for x in batch_tup]}")
            input_t = batch_tup[0]
            # Try to find a tensor that could be the label
            label_t = None
            for item in batch_tup[1:]:
                if isinstance(item, torch.Tensor) and item.dtype != torch.float32:
                    label_t = item.float()
                    break
            if label_t is None:
                # Fallback to the second item and hope for the best
                label_t = batch_tup[1].float() if len(batch_tup) > 1 else torch.zeros_like(input_t[:, 0, 0, 0])
        
        # Make sure label_t is float32
        label_t = label_t.to(torch.float32)
        
        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)
        log.debug(f'input_g: {input_g.shape}, {input_g.dtype}')
        log.debug(f'label_g: {label_g.shape}, {label_g.dtype}')
        
        prob_g = self.model(input_g)
        prob_g = prob_g.squeeze(dim=-1)  # [batch_size]
        pred_g = (prob_g >= self.threshold).float()
        
        # Ensure label tensor has the right shape
        label_g = label_g.view(-1)  # [batch_size]
        
        # Compute loss
        loss_g = self.loss_fn(prob_g, label_g)
        
        # Calculate batch-specific weights
        batch_pos = (label_g > 0.5).sum().item()
        batch_neg = label_g.size(0) - batch_pos
        
        if batch_neg > 0:
            pos_weight = 1.0
            # Adjust weight based on batch composition, with a cap
            neg_weight = min(batch_pos / batch_neg * 3.0, 10.0)
        else:
            pos_weight = 1.0
            neg_weight = 5.0  # Default if no negatives in batch
        
        # Apply weights to individual samples
        sample_weights = torch.ones_like(label_g)
        sample_weights[label_g < 0.5] = neg_weight
        
        # Apply weighted loss
        loss_g = self.loss_fn(prob_g, label_g) * sample_weights
        loss_mean = loss_g.mean()
        
        # Store metrics
        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_g.size(0)
        
        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = label_g.detach()
        metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = pred_g.detach()
        metrics_g[METRICS_PROB_NDX, start_ndx:end_ndx] = prob_g.detach()
        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss_g.detach()
        
        return loss_mean


# TensorBoard Logger
class TensorBoardLogger:
    def __init__(self, tb_prefix, time_str, comment=''):
        self.log_dir = os.path.join('runs', tb_prefix, time_str)
        self.trn_writer = SummaryWriter(log_dir=self.log_dir + f'-trn_cls-{comment}')
        self.val_writer = SummaryWriter(log_dir=self.log_dir + f'-val_cls-{comment}')
    
    def log_metrics(self, mode_str, epoch, metrics, sample_count):
        writer = getattr(self, f'{mode_str}_writer')
        
        # Log basic metrics
        writer.add_scalar(f'loss/all', metrics[METRICS_LOSS_NDX].mean(), sample_count)
        
        # Compute and log classification metrics
        y_true = metrics[METRICS_LABEL_NDX].numpy()
        y_pred = metrics[METRICS_PRED_NDX].numpy()
        y_prob = metrics[METRICS_PROB_NDX].numpy()
        
        try:
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            writer.add_scalar(f'metrics/accuracy', acc, sample_count)
            writer.add_scalar(f'metrics/precision', prec, sample_count)
            writer.add_scalar(f'metrics/recall', rec, sample_count)
            writer.add_scalar(f'metrics/f1', f1, sample_count)
            
            # Only compute AUC if we have both classes
            if len(np.unique(y_true)) > 1:
                auc = roc_auc_score(y_true, y_prob)
                writer.add_scalar(f'metrics/auc', auc, sample_count)
        except Exception as e:
            log.warning(f"Failed to compute some metrics: {e}")
    
    def close(self):
        self.trn_writer.close()
        self.val_writer.close()


class LogisticRegressionApp:
    def __init__(self, sys_argv=None, input_df=None, output_df=None):
        """
        Initialize with optional pre-loaded data to simplify the sequence.
        """
        # Parse arguments first
        self.config = Config().parse_args(sys_argv)
        
        # Set device
        if not self.config.DEBUG:
            self.use_cuda = torch.cuda.is_available()
            self.device = torch.device("cuda" if self.use_cuda else "cpu")
        else:
            self.use_cuda = False
            self.device = "cpu"
        
        # Use pre-loaded data if provided, otherwise load it
        if input_df is not None and output_df is not None:
            self.input_df = input_df
            self.output_df = output_df
        else:
            self._load_data()
        
        # Add threshold optimization config
        self.config.threshold_optimization = True
        self.config.pauc_fpr_limit = 0.1  # For pAUC calculation
        
        # Data integrity check
        try:
            log.info("Checking data integrity...")
            self.check_data_integrity()
            log.info("Data integrity check passed")
        except Exception as e:
            log.warning(f"Data integrity check failed: {e}")
        
        # Set folder path
        self.folder = self.config.folder
        
        # Set up train/validation subjects directly
        log.info("Setting up train/validation subjects...")
        # Use existing columns if specified, otherwise use fixed ratio
        if self.config.use_train_validation_cols:
            # Use existing training/validation flags from the DataFrame
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
        
        # Set up model
        log.info("Setting up model...")
        self._setup_model()
        log.info("Model set up successfully")
    
        # Initialize data handler
        log.info("Initializing DataHandler...")
        self.data_handler = DataHandler(
            self.input_df, self.output_df, self.use_cuda, 
            self.config.batch_size, self.config.num_workers
        )
        # Set any additional properties
        if hasattr(self.config, 'augment_minority'):
            self.data_handler.augment_minority = self.config.augment_minority
        log.info("DataHandler initialized")
        
        # Set up dataloaders using DataHandler
        log.info("Setting up dataloaders...")
        self.train_dl = self.data_handler.init_dl(self.folder, self.train_subjects)
        self.val_dl = self.data_handler.init_dl(self.folder, self.val_subjects, is_val_set=True)
        log.info("Dataloaders set up successfully")
        
        # Set up optimizer and scheduler
        log.info("Setting up optimizer...")
        self._setup_optimizer()
        log.info("Setting up scheduler...")
        self._setup_scheduler()
        log.info("Optimizer and scheduler set up successfully")
    
        # Set up TensorBoard logger
        log.info("Setting up TensorBoard logger...")
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.tb_logger = TensorBoardLogger(
            self.config.tb_prefix, self.time_str, self.config.comment
        )
        log.info("TensorBoard logger set up successfully")
        
        log.info("LogisticRegressionApp initialized successfully")

    def train(self, fold_idx=None):
        """Modified train method with threshold optimization"""
        if self.config.threshold_optimization:
            metrics = self.train_with_threshold_optimization(fold_idx=fold_idx)
        else:
            # Use standard training method
            metrics = self._standard_train(fold_idx=fold_idx)
        
        # After training completes, plot ROC curve
        self.plot_roc_curve(fold_idx=fold_idx)
       
        return metrics


    def _standard_train(self):
        """
        Execute the standard training loop over multiple epochs.
        """
        log.info("Starting training process...")
            
        # Ensure we have data to train on
        if len(self.train_dl.dataset) == 0:
            raise ValueError("Training dataset is empty. Cannot proceed with training.")
            
        if len(self.val_dl.dataset) == 0:
            raise ValueError("Validation dataset is empty. Cannot proceed with training.")

        try:        
            # Initialize the trainer with model, optimizer, and device
            trainer = LogisticRegressionTrainer(
                self.model, 
                self.optimizer, 
                self.device, 
                threshold=self.config.threshold
            )
            
            # Track the best model
            best_val_loss = float('inf')
            best_val_acc = 0.0
            best_epoch = 0
            
            # Train for the specified number of epochs
            for epoch in range(1, self.config.epochs + 1):
                log.info(f"Epoch {epoch}/{self.config.epochs}")
                
                # Training phase
                trn_metrics = trainer.train_epoch(epoch, self.train_dl)
                trn_loss = trn_metrics[METRICS_LOSS_NDX].mean().item()
                
                # Validation phase
                val_metrics = trainer.validate_epoch(epoch, self.val_dl)
                val_loss = val_metrics[METRICS_LOSS_NDX].mean().item()
                
                # Calculate validation accuracy
                val_acc = accuracy_score(
                    val_metrics[METRICS_LABEL_NDX].numpy(), 
                    val_metrics[METRICS_PRED_NDX].numpy()
                )
            
                # Log metrics to TensorBoard
                self.tb_logger.log_metrics('trn', epoch, trn_metrics, trainer.total_samples)
                self.tb_logger.log_metrics('val', epoch, val_metrics, trainer.total_samples)
                
                # Update learning rate scheduler
                if self.config.scheduler == 'plateau':
                    self.scheduler.step(val_loss)
                elif self.config.scheduler != 'onecycle':
                    self.scheduler.step()
                
                # Check if this is the best model so far
                is_best = False
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_val_loss = val_loss
                    best_epoch = epoch
                    is_best = True
                    
                    # Save the best model
                    if self.config.model_save_location:
                        log.info(f"New best validation accuracy: {best_val_acc:.4f}, saving model")
                        self.save_model(self.config.model_save_location)
                
                # Log epoch summary
                log.info(f"Epoch {epoch}/{self.config.epochs} - "
                        f"Train Loss: {trn_loss:.4f}, "
                        f"Val Loss: {val_loss:.4f}, "
                        f"Val Accuracy: {val_acc:.4f}, "
                        f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                        f"{' (BEST)' if is_best else ''}")
            
            log.info(f"Training completed. Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
            
            # Save final model if no best model was saved during training
            if not self.config.model_save_location:
                final_model_path = f'./logistic_model_final-{self.time_str}.pt'
                self.save_model(final_model_path)
                log.info(f"Final model saved to {final_model_path}")
                
            # Generate comprehensive summary statistics
            self.generate_summary_statistics()
            
            # Close tensorboard logger
            self.tb_logger.close()
        except Exception as e:
            log.error(f"Error during training: {e}")
            raise

    def train_with_threshold_optimization(self, fold_idx=None):
        """
        Execute the training loop with threshold optimization after each epoch.
        """
        log.info("Starting training with threshold optimization...")
        
        # Initialize the trainer
        trainer = LogisticRegressionTrainer(
            self.model, 
            self.optimizer, 
            self.device, 
            threshold=self.config.threshold  # Initial threshold
        )
        
        # Track best model
        best_pauc = 0.0
        best_threshold = self.config.threshold
        best_epoch = 0
        best_metrics = {}  # Store the best metrics

        # Arrays to store metrics for plotting
        epoch_metrics = {
            'thresholds': [],
            'paucs': [],
            'f1_scores': [],
            'accuracies': []
        }
        
        for epoch in range(1, self.config.epochs + 1):
            log.info(f"Epoch {epoch}/{self.config.epochs}")
            
            # Training phase
            trn_metrics = trainer.train_epoch(epoch, self.train_dl)
            
            # Validation phase  
            val_metrics = trainer.validate_epoch(epoch, self.val_dl)
            
            # Find optimal threshold for pAUC after validation
            optimal_threshold, current_pauc, threshold_metrics = trainer.find_optimal_threshold_for_pauc(self.val_dl)
            
            # Update trainer's threshold for next epoch
            trainer.threshold = optimal_threshold
            
            # Log metrics
            log.info(f"Epoch {epoch} Results:")
            log.info(f"  Optimal Threshold: {optimal_threshold:.4f}")
            log.info(f"  pAUC: {current_pauc:.4f}")
            log.info(f"  Accuracy: {threshold_metrics.get('accuracy', 0):.4f}")
            log.info(f"  F1 Score: {threshold_metrics.get('f1', 0):.4f}")
            log.info(f"  Sensitivity: {threshold_metrics.get('sensitivity', 0):.4f}")
            log.info(f"  Specificity: {threshold_metrics.get('specificity', 0):.4f}")
        
            # Track metrics
            epoch_metrics['thresholds'].append(optimal_threshold)
            epoch_metrics['paucs'].append(current_pauc)
            epoch_metrics['f1_scores'].append(threshold_metrics.get('f1', 0))
            epoch_metrics['accuracies'].append(threshold_metrics.get('accuracy', 0))
            
            # Save best model based on pAUC
            if current_pauc > best_pauc:
                best_pauc = current_pauc
                best_threshold = optimal_threshold
                best_epoch = epoch
                best_metrics = threshold_metrics.copy()  # Save the best metrics
                
            # Update learning rate scheduler
            if self.config.scheduler == 'plateau':
                self.scheduler.step(-current_pauc)  # Use negative pAUC to maximize
            elif self.config.scheduler != 'onecycle':
                self.scheduler.step()
                
        if self.config.model_save_location:
            base_path = os.path.splitext(self.config.model_save_location)[0]
            if fold_idx is not None:
                save_path = f"{base_path}_fold{fold_idx+1}_best_pauc.pt"
                threshold_path = f"{base_path}_fold{fold_idx+1}_threshold_info.json"
            else:
                save_path = f"{base_path}_best_pauc.pt"
                threshold_path = f"{base_path}_threshold_info.json"
            
            self.save_model(save_path)

            # Save threshold information
            threshold_info = {
                'threshold': float(best_threshold),  # Use best values, not current
                'pauc': float(best_pauc),           
                'epoch': int(best_epoch),           
                'metrics': {k: float(v) if isinstance(v, np.floating) else v 
                            for k, v in best_metrics.items()}
            }

            with open(threshold_path, 'w') as f:
                json.dump(threshold_info, f, indent=4)
            
            log.info(f"New best pAUC: {best_pauc:.4f}, saved to {save_path}")        
        
        log.info(f"Training completed. Best pAUC: {best_pauc:.4f} at epoch {best_epoch}")
        log.info(f"Best threshold: {best_threshold:.4f}")

        # Plot threshold optimization results with fold-specific name
        self.plot_threshold_optimization_results(epoch_metrics, fold_idx=fold_idx)

        # Return all metrics from the best epoch
        return_metrics = {
            'threshold': best_threshold,
            'pAUC': best_pauc,
            'epoch': best_epoch,
            'accuracy': best_metrics.get('accuracy', 0),
            'precision': best_metrics.get('precision', 0),
            'recall': best_metrics.get('recall', 0),
            'f1': best_metrics.get('f1', 0),
            'sensitivity': best_metrics.get('sensitivity', 0),
            'specificity': best_metrics.get('specificity', 0),
            'ppv': best_metrics.get('ppv', 0),
            'npv': best_metrics.get('npv', 0),
            'auc': best_metrics.get('auc', 0)
        }
        
        return return_metrics

    def plot_threshold_optimization_results(self, epoch_metrics, fold_idx=None):
        """Plot threshold optimization results over epochs"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        epochs = list(range(1, len(epoch_metrics['thresholds']) + 1))
        
        # Plot threshold evolution
        axes[0, 0].plot(epochs, epoch_metrics['thresholds'], 'b-', marker='o')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Optimal Threshold')
        axes[0, 0].set_title('Threshold Evolution')
        axes[0, 0].grid(True)

        # Plot pAUC evolution
        axes[0, 1].plot(epochs, epoch_metrics['paucs'], 'r-', marker='o')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('pAUC')
        axes[0, 1].set_title('pAUC Evolution')
        axes[0, 1].grid(True)
        
        # Plot F1 score evolution
        axes[1, 0].plot(epochs, epoch_metrics['f1_scores'], 'g-', marker='o')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('F1 Score Evolution')
        axes[1, 0].grid(True)
        
        # Plot accuracy evolution
        axes[1, 1].plot(epochs, epoch_metrics['accuracies'], 'm-', marker='o')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Accuracy Evolution')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        if self.config.plot_location:
            base_plot_path = os.path.splitext(self.config.plot_location)[0]
            if fold_idx is not None:
                plot_path = f"{base_plot_path}_threshold_optimization_fold{fold_idx+1}.png"
            else:
                plot_path = f"{base_plot_path}_threshold_optimization.png"
            
            # CREATE THE DIRECTORY IF IT DOESN'T EXIST
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            
            plt.savefig(plot_path)
            log.info(f"Threshold optimization plot saved to {plot_path}")
        
        plt.close()  # Close the figure to free memory

    def _setup_scheduler(self):
        """
        Set up the learning rate scheduler based on the configuration.
        """
        log.info(f"Setting up {self.config.scheduler} scheduler...")
        
        if self.config.scheduler == 'step':
            self.scheduler = StepLR(
                self.optimizer, 
                step_size=30, 
                gamma=0.1
            )
            log.info(f"Step scheduler with step_size=30, gamma=0.1")
            
        elif self.config.scheduler == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config.epochs, 
                eta_min=0
            )
            log.info(f"Cosine scheduler with T_max={self.config.epochs}")
            
        elif self.config.scheduler == 'onecycle':
            # Calculate total steps based on dataloader length and epochs
            total_steps = len(self.train_dl) * self.config.epochs
            
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config.lr * 10,  # Max LR is 10x base LR
                total_steps=total_steps,
                pct_start=0.3  # Spend 30% of time in warmup
            )
            log.info(f"OneCycle scheduler with total_steps={total_steps}, pct_start=0.3")
            
        else:  # 'plateau' (default)
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, 
                mode='min',  # Reduce LR when the validation loss stops decreasing
                factor=0.1,  # Multiply LR by this factor
                patience=10
            )
            log.info(f"ReduceLROnPlateau scheduler with factor=0.1, patience=10")
        
        log.info(f"Scheduler initialized: {type(self.scheduler).__name__}")

    def _setup_optimizer(self):
        # Verify that the model has parameters before creating optimizer
        if not any(p.requires_grad for p in self.model.parameters()):
            log.error("No parameters in the model require gradients!")
            # Add a dummy parameter to prevent optimizer error
            self.model.dummy_param = nn.Parameter(torch.zeros(1, requires_grad=True))
        
        # Get list of parameters that require gradients
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Log parameter information
        log.info(f"Setting up optimizer with {len(params)} parameter groups")
        
        if self.config.optimizer.lower() == 'adam':
            self.optimizer = Adam(
                params,
                lr=self.config.lr,
                weight_decay=self.config.weight_decay
            )
        else:
            self.optimizer = SGD(
                params,
                lr=self.config.lr, 
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        
        log.info(f"Optimizer initialized: {type(self.optimizer).__name__}")

    def _setup_model(self):        
        """
        Set up the model based on configuration.
        """
        model_type = self.config.model_type
        debug_mode = self.config.DEBUG
        
        log.info(f"Setting up {model_type} model")
        
        try:
            # Create model based on type
            if model_type == 'simple':
                # Simple model for testing or baseline performance
                input_dim = len(self.config.features)
                log.info(f"Creating simple model with {input_dim} input features")
                self.model = SimpleMRIModel(input_dim=input_dim, debug=debug_mode)
            
            elif model_type == 'conv':
                # Import the convolutional model
                try:
                    log.info("Importing MRI logistic regression model")
                    self.model = get_mri_logistic_regression_model(model_type='conv', debug=debug_mode)
                except ImportError as e:
                    log.error(f"Failed to import get_mri_logistic_regression_model: {e}")
                    log.warning("Falling back to SimpleMRIModel due to import error")
                    input_dim = len(self.config.features)
                    self.model = SimpleMRIModel(input_dim=input_dim, debug=debug_mode)
            
            elif model_type in ['resnet3d', 'dense3d', 'efficientnet3d']:
                try:
                    log.info(f"Importing advanced model: {model_type}")
                    from dcan.models.advanced_mri_models import get_advanced_mri_model
                    self.model = get_advanced_mri_model(model_type=model_type, debug=debug_mode)
                except ImportError as e:
                    log.error(f"Failed to import advanced model modules: {e}")
                    raise ValueError(f"Advanced model '{model_type}' requires additional dependencies that are not available")
            
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        
            # Validate model structure
            if self.model is None:
                raise ValueError("Model initialization failed, returned None")
                
            # Move model to the appropriate device
            self.model = self.model.to(self.device)
            log.info(f"Model moved to {self.device}")
            
            # Log model parameter count and structure
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            log.info(f"Model has {total_params:,} total parameters ({trainable_params:,} trainable)")
            
            # Ensure model is in training mode
            self.model.train()
            
        except Exception as e:
            log.error(f"Error creating model: {e}")
            raise
                
        log.info(f"Model initialization complete: {model_type}")

    def check_data_integrity(self):
        """
        Performs comprehensive checks to ensure data integrity before training.
        """
        log.info("Checking data integrity...")
        
        # Check for empty dataset
        if self.input_df.empty:
            raise ValueError("Input DataFrame is empty. Cannot proceed with training.")
        
        log.info("Data integrity check completed.")

    def _load_data(self):
        """
        Load and validate input data from CSV files.
        """
        log.info(f"Loading data from {self.config.csv_input_file}")
        
        try:
            # Validate file existence
            if not os.path.exists(self.config.csv_input_file):
                raise FileNotFoundError(f"Input file not found: {self.config.csv_input_file}")
            
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
            log.info(f"Features: {self.config.features}")
            log.info(f"Target: {self.config.target}")
            log.info(f"Class distribution: {self.input_df[self.config.target].value_counts().to_dict()}")
        
        except Exception as e:
            log.error(f"Error loading data: {e}")
            raise

    def _normalize_features(self):
        """
        Normalize feature columns to have zero mean and unit variance.
        """
        from sklearn.preprocessing import StandardScaler
        
        log.info("Normalizing feature columns...")
        
        scaler = StandardScaler()
        
        # Store the original values before normalization
        self.original_feature_values = self.input_df[self.config.features].copy()
        
        # Normalize only the configured feature columns
        self.input_df[self.config.features] = scaler.fit_transform(
            self.input_df[self.config.features]
        )
        
        # Save scaler for later use
        self.feature_scaler = scaler
        
        log.info("Feature normalization complete")

    def run_cross_validation(self, k=5):
        """
        Run k-fold cross-validation on the dataset.
        """
        log.info(f"Starting {k}-fold cross-validation")
        
        # Get all unique subjects
        all_subjects = list(set(self.input_df['anonymized_subject_id'].tolist()))
        
        # Stratify subjects based on whether they have ALD
        subjects_with_ald = []
        subjects_without_ald = []
        
        for subject in all_subjects:
            subject_rows = self.input_df[self.input_df['anonymized_subject_id'] == subject]
            max_loes = subject_rows['loes-score'].max()
            if max_loes > self.config.threshold:
                subjects_with_ald.append(subject)
            else:
                subjects_without_ald.append(subject)
        
        log.info(f"Total subjects: {len(all_subjects)} ({len(subjects_with_ald)} with ALD, {len(subjects_without_ald)} without ALD)")
    
        # Create stratified folds
        random.shuffle(subjects_with_ald)
        random.shuffle(subjects_without_ald)
        
        ald_folds = self._create_folds(subjects_with_ald, k)
        no_ald_folds = self._create_folds(subjects_without_ald, k)

        # Metrics to track across folds
        fold_metrics = {
            'accuracy': [],
            'precision': [],
            'ppv': [],
            'recall': [],
            'f1': [],
            'auc': [],
            'sensitivity': [],
            'specificity': [],
            'pAUC': []
        }
        
        # Save path for the best model across all folds
        best_model_path = self.config.model_save_location
        best_val_pauc = 0.0
    
        # Run training for each fold
        for fold in range(k):
            log.info(f"\n{'='*40}")
            log.info(f"FOLD {fold+1}/{k}")
            log.info(f"{'='*40}")
            
            # Create train/validation split for this fold
            val_subjects_ald = ald_folds[fold]
            val_subjects_no_ald = no_ald_folds[fold]
            
            train_subjects_ald = [s for i, fold_subjects in enumerate(ald_folds) for s in fold_subjects if i != fold]
            train_subjects_no_ald = [s for i, fold_subjects in enumerate(no_ald_folds) for s in fold_subjects if i != fold]
            
            self.train_subjects = train_subjects_ald + train_subjects_no_ald
            self.val_subjects = val_subjects_ald + val_subjects_no_ald
        
            log.info(f"Train set: {len(self.train_subjects)} subjects ({len(train_subjects_ald)} with ALD, {len(train_subjects_no_ald)} without ALD)")
            log.info(f"Val set: {len(self.val_subjects)} subjects ({len(val_subjects_ald)} with ALD, {len(val_subjects_no_ald)} without ALD)")
            
            # Set up dataloaders for this fold
            self.train_dl = self.data_handler.init_dl(self.folder, self.train_subjects)
            self.val_dl = self.data_handler.init_dl(self.folder, self.val_subjects, is_val_set=True)
            
            # Initialize a new model for this fold
            self._setup_model()
            self._setup_optimizer()
            self._setup_scheduler()
            
            # Train model for this fold
            fold_results = self._train_fold(fold)
            
            # Track metrics
            for metric in fold_metrics:
                if metric in fold_results:
                    fold_metrics[metric].append(fold_results[metric])
            
            # Save best model across folds
            if fold_results.get('pAUC', 0) > best_val_pauc:
                best_val_pauc = fold_results.get('pAUC', 0)
                fold_model_path = f"{os.path.splitext(best_model_path)[0]}_fold{fold+1}.pt"
                self.save_model(fold_model_path)
                log.info(f"New best model saved to {fold_model_path}")
        
        # Print cross-validation summary
        log.info("\n" + "="*80)
        log.info("CROSS-VALIDATION RESULTS")
        log.info("="*80)

        for metric, values in fold_metrics.items():
            if values:
                mean_value = np.mean(values)
                std_value = np.std(values)
                log.info(f"{metric.capitalize()}: {mean_value:.4f} ± {std_value:.4f}")
        
        log.info("="*80)

    def _create_folds(self, subjects, k):
        """
        Split list of subjects into k approximately equal folds.
        """
        folds = [[] for _ in range(k)]
        for i, subject in enumerate(subjects):
            folds[i % k].append(subject)
        return folds
    
    def _train_fold(self, fold_idx):
        # Set up model, optimizer, etc.
        self._setup_model()
        self._setup_optimizer()
        self._setup_scheduler()
        
        # Pass fold index to train method
        return self.train(fold_idx=fold_idx)

    def generate_summary_statistics(self):
        """
        Generate and display comprehensive summary statistics after training.
        """
        log.info("\n" + "="*80)
        log.info("TRAINING SUMMARY STATISTICS")
        log.info("="*80)
        
        # 1. Model architecture information
        log.info("\nMODEL INFORMATION:")
        log.info(f"Model type: {self.config.model_type}")
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        log.info(f"Total parameters: {total_params:,}")
        log.info(f"Trainable parameters: {trainable_params:,}")
        
        # 2. Training configuration
        log.info("\nTRAINING CONFIGURATION:")
        log.info(f"Epochs: {self.config.epochs}")
        log.info(f"Batch size: {self.config.batch_size}")
        log.info(f"Optimizer: {self.config.optimizer} (LR: {self.config.lr})")
        log.info(f"Scheduler: {self.config.scheduler}")
        log.info(f"Weight decay: {self.config.weight_decay}")
        log.info(f"Classification threshold: {self.config.threshold}")
    
        # 3. Dataset information
        log.info("\nDATASET INFORMATION:")
        train_subjects_count = len(self.train_subjects)
        val_subjects_count = len(self.val_subjects)
        total_subjects = train_subjects_count + val_subjects_count
        
        train_rows = self.input_df[self.input_df['anonymized_subject_id'].isin(self.train_subjects)]
        val_rows = self.input_df[self.input_df['anonymized_subject_id'].isin(self.val_subjects)]
        
        train_scans = len(train_rows)
        val_scans = len(val_rows)
        total_scans = train_scans + val_scans
        
        log.info(f"Subjects: {total_subjects} total ({train_subjects_count} training, {val_subjects_count} validation)")
        log.info(f"Scans: {total_scans} total ({train_scans} training, {val_scans} validation)")
    
        # 4. Class distribution
        try:
            train_positive = (train_rows['loes-score'] > self.config.threshold).sum()
            train_negative = len(train_rows) - train_positive
            val_positive = (val_rows['loes-score'] > self.config.threshold).sum()
            val_negative = len(val_rows) - val_positive
            
            log.info("\nCLASS DISTRIBUTION:")
            log.info(f"Training: {train_positive} positive ({train_positive/train_scans*100:.1f}%), " +
                    f"{train_negative} negative ({train_negative/train_scans*100:.1f}%)")
            log.info(f"Validation: {val_positive} positive ({val_positive/val_scans*100:.1f}%), " +
                    f"{val_negative} negative ({val_negative/val_scans*100:.1f}%)")
        except:
            log.warning("Could not compute class distribution")
    
        # 5. Final model performance metrics
        log.info("\nFINAL MODEL PERFORMANCE (Validation Set):")
        
        try:
            # Make predictions on validation set
            self.model.eval()
            val_preds = []
            val_labels = []
            val_probs = []
            
            with torch.no_grad():
                for batch_tup in self.val_dl:
                    input_t, label_t, _, _ = batch_tup
                    label = [1.0 if l_t.item() > self.config.threshold else 0.0 for l_t in label_t]
                    val_labels.extend(label)
                    
                    input_g = input_t.to(self.device)
                    probs = self.model(input_g).squeeze().cpu().numpy()
                    preds = (probs >= self.config.threshold).astype(float)
                    
                    val_probs.extend(probs)
                    val_preds.extend(preds)
            
            # Calculate metrics
            acc = accuracy_score(val_labels, val_preds)
            prec = precision_score(val_labels, val_preds, zero_division=0)
            rec = recall_score(val_labels, val_preds, zero_division=0)
            f1 = f1_score(val_labels, val_preds, zero_division=0)
            
            log.info(f"Accuracy: {acc:.4f}")
            log.info(f"Precision: {prec:.4f}")
            log.info(f"Recall: {rec:.4f}")
            log.info(f"F1 Score: {f1:.4f}")
        
            # Compute AUC if we have both classes
            if len(np.unique(val_labels)) > 1:
                p_auc = score(val_labels, val_probs)
                log.info(f"pAUC: {p_auc:.4f}")
                auc_score = roc_auc_score(val_labels, val_probs)
                log.info(f"AUC: {auc_score:.4f}")
            
            # Confusion matrix
            cm = confusion_matrix(val_labels, val_preds)
            log.info("\nConfusion Matrix:")
            log.info("                  Predicted Negative    Predicted Positive")
            log.info(f"Actual Negative       {cm[0][0]:8d}              {cm[0][1]:8d}")
            log.info(f"Actual Positive       {cm[1][0]:8d}              {cm[1][1]:8d}")
            
            # Calculate specificity and sensitivity
            sensitivity = rec  # Same as recall
            if cm[0][0] + cm[0][1] > 0:
                specificity = cm[0][0] / (cm[0][0] + cm[0][1])
            else:
                specificity = 0
            log.info(f"\nSensitivity: {sensitivity:.4f}")
            log.info(f"Specificity: {specificity:.4f}")
            
        except Exception as e:
            log.error(f"Error computing final model performance: {e}")
        
        log.info("="*80)

    def save_model(self, path):
        """Save the trained model to disk"""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save(self.model.state_dict(), path)
        log.info(f"Model saved to {path}")

    def plot_roc_curve(self, val_dl=None, save_path=None, fold_idx=None):
        """
        Plot ROC curve for the trained model on validation set
        
        Args:
            val_dl: Validation dataloader (uses self.val_dl if None)
            save_path: Path to save the plot (uses config.plot_location if None)
            fold_idx: Fold index for cross-validation (optional)
        """
        log.info("Plotting ROC curve...")
        
        if val_dl is None:
            val_dl = self.val_dl
            
        # Set model to evaluation mode
        self.model.eval()
        all_probs = []
        all_labels = []
        
        # Collect all predictions and labels
        with torch.no_grad():
            for batch_tup in val_dl:
                # Handle different batch tuple formats
                if len(batch_tup) == 4:
                    input_t, label_t, _, _ = batch_tup
                    labels = [1.0 if l_t.item() > self.config.threshold else 0.0 for l_t in label_t]
                    all_labels.extend(labels)
                elif len(batch_tup) == 5:
                    input_t, _, has_ald_t, _, _ = batch_tup
                    all_labels.extend(has_ald_t.float().cpu().numpy())
                else:
                    # Handle other formats
                    input_t = batch_tup[0]
                    label_t = batch_tup[1]
                    labels = label_t.float().cpu().numpy()
                    all_labels.extend(labels)
                
                # Get model predictions
                input_g = input_t.to(self.device)
                probs = self.model(input_g).squeeze().cpu().numpy()
                
                # Handle single sample case
                if np.ndim(probs) == 0:
                    probs = np.array([probs])
                
                all_probs.extend(probs)
        
        # Convert to numpy arrays
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        
        # Calculate partial AUC (pAUC) at FPR <= 0.1
        pauc_fpr_limit = 0.1
        limited_fpr = fpr[fpr <= pauc_fpr_limit]
        limited_tpr = tpr[:len(limited_fpr)]
        
        if len(limited_fpr) > 1:
            pAUC = auc(limited_fpr, limited_tpr) / pauc_fpr_limit
        else:
            pAUC = 0.0
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        
        # Plot ROC curve
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        
        # Mark current threshold point
        current_threshold_idx = np.argmin(np.abs(thresholds - self.config.threshold))
        plt.plot(fpr[current_threshold_idx], tpr[current_threshold_idx], 
                'ro', markersize=10, 
                label=f'Current threshold ({self.config.threshold:.3f})')
        
        # Find and mark optimal threshold (Youden's J statistic)
        youden_j = tpr - fpr
        optimal_idx = np.argmax(youden_j)
        optimal_threshold = thresholds[optimal_idx]
        plt.plot(fpr[optimal_idx], tpr[optimal_idx], 
                'go', markersize=10, 
                label=f'Optimal threshold ({optimal_threshold:.3f})')
        
        # Customize plot
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        
        if fold_idx is not None:
            plt.title(f'ROC Curve - Fold {fold_idx + 1}')
        else:
            plt.title('Receiver Operating Characteristic (ROC) Curve')
        
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Add metrics text box
        metrics_text = f'Metrics at current threshold ({self.config.threshold:.3f}):\n'
        
        # Calculate metrics at current threshold
        y_pred = (all_probs >= self.config.threshold).astype(int)
        
        from sklearn.metrics import (accuracy_score, precision_score, 
                                    recall_score, f1_score, confusion_matrix)
        
        accuracy = accuracy_score(all_labels, y_pred)
        precision = precision_score(all_labels, y_pred, zero_division=0)
        recall = recall_score(all_labels, y_pred, zero_division=0)
        f1 = f1_score(all_labels, y_pred, zero_division=0)
        
        # Calculate specificity
        cm = confusion_matrix(all_labels, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            specificity = 0.0
        
        metrics_text += f'Accuracy: {accuracy:.3f}\n'
        metrics_text += f'Precision: {precision:.3f}\n'
        metrics_text += f'Recall: {recall:.3f}\n'
        metrics_text += f'Specificity: {specificity:.3f}\n'
        metrics_text += f'F1 Score: {f1:.3f}'
        
        # Add text box
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.55, 0.15, metrics_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            if self.config.plot_location:
                base_path = os.path.splitext(self.config.plot_location)[0]
                if fold_idx is not None:
                    save_path = f"{base_path}_roc_fold{fold_idx+1}.png"
                else:
                    save_path = f"{base_path}_roc.png"
            else:
                save_path = f"roc_curve_{self.time_str}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        log.info(f"ROC curve saved to {save_path}")
        
        # Also save as PDF for publication quality
        pdf_path = save_path.rsplit('.', 1)[0] + '.pdf'
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        log.info(f"ROC curve saved to {pdf_path}")
        
        plt.close()
        
        return roc_auc, pAUC
    
    def plot_multiple_roc_curves(self, models_data, save_path=None):
        """
        Plot multiple ROC curves (e.g., for different folds or models) on the same plot
        
        Args:
            models_data: List of tuples (name, all_labels, all_probs)
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
        
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        aucs = []
        
        for i, (name, all_labels, all_probs) in enumerate(models_data):
            fpr, tpr, _ = roc_curve(all_labels, all_probs)
            tprs_interp = np.interp(mean_fpr, fpr, tpr)
            tprs_interp[0] = 0.0
            tprs.append(tprs_interp)
            
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            
            plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2, 
                    label=f'{name} (AUC = {roc_auc:.3f})')
        
        # Plot mean ROC
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        
        plt.plot(mean_fpr, mean_tpr, color='navy', lw=3,
                label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})',
                alpha=0.8)
        
        # Plot standard deviation
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', 
                        alpha=0.2, label=r'± 1 std. dev.')
        
        # Plot diagonal
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', 
                label='Random Classifier', alpha=0.8)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = f"multiple_roc_curves_{self.time_str}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.rsplit('.', 1)[0] + '.pdf', dpi=300, bbox_inches='tight')
        
        plt.close()
        
        log.info(f"Multiple ROC curves saved to {save_path}")
    
    # Modified cross-validation method to include ROC plotting
    def run_cross_validation_with_roc(self, k=5):
        """
        Run k-fold cross-validation and plot ROC curves
        """
        log.info(f"Starting {k}-fold cross-validation with ROC curves")
        
        # Store data for combined ROC plot
        all_fold_data = []
        
        # Get all unique subjects and create folds as before
        all_subjects = list(set(self.input_df['anonymized_subject_id'].tolist()))
        
        # Stratify subjects
        subjects_with_ald = []
        subjects_without_ald = []
        
        for subject in all_subjects:
            subject_rows = self.input_df[self.input_df['anonymized_subject_id'] == subject]
            max_loes = subject_rows['loes-score'].max()
            if max_loes > self.config.threshold:
                subjects_with_ald.append(subject)
            else:
                subjects_without_ald.append(subject)
        
        # Create folds
        random.shuffle(subjects_with_ald)
        random.shuffle(subjects_without_ald)
        
        ald_folds = self._create_folds(subjects_with_ald, k)
        no_ald_folds = self._create_folds(subjects_without_ald, k)
        
        # Run training for each fold
        for fold in range(k):
            log.info(f"\n{'='*40}")
            log.info(f"FOLD {fold+1}/{k}")
            log.info(f"{'='*40}")
            
            # Set up train/validation split for this fold
            val_subjects_ald = ald_folds[fold]
            val_subjects_no_ald = no_ald_folds[fold]
            
            train_subjects_ald = [s for i, fold_subjects in enumerate(ald_folds) 
                                 for s in fold_subjects if i != fold]
            train_subjects_no_ald = [s for i, fold_subjects in enumerate(no_ald_folds) 
                                    for s in fold_subjects if i != fold]
            
            self.train_subjects = train_subjects_ald + train_subjects_no_ald
            self.val_subjects = val_subjects_ald + val_subjects_no_ald
            
            # Set up dataloaders
            self.train_dl = self.data_handler.init_dl(self.folder, self.train_subjects)
            self.val_dl = self.data_handler.init_dl(self.folder, self.val_subjects, is_val_set=True)
            
            # Initialize model and train
            self._setup_model()
            self._setup_optimizer()
            self._setup_scheduler()
            
            # Train model for this fold
            fold_results = self._train_fold(fold)
            
            # Plot ROC curve for this fold
            auc_score, pauc_score = self.plot_roc_curve(fold_idx=fold)
            
            # Collect predictions for combined plot
            self.model.eval()
            all_probs = []
            all_labels = []
            
            with torch.no_grad():
                for batch_tup in self.val_dl:
                    if len(batch_tup) == 4:
                        input_t, label_t, _, _ = batch_tup
                        labels = [1.0 if l_t.item() > self.config.threshold else 0.0 for l_t in label_t]
                        all_labels.extend(labels)
                    elif len(batch_tup) == 5:
                        input_t, _, has_ald_t, _, _ = batch_tup
                        all_labels.extend(has_ald_t.float().cpu().numpy())
                    
                    input_g = input_t.to(self.device)
                    probs = self.model(input_g).squeeze().cpu().numpy()
                    
                    if np.ndim(probs) == 0:
                        probs = np.array([probs])
                    
                    all_probs.extend(probs)
            
            all_fold_data.append((f'Fold {fold+1}', all_labels, all_probs))
        
        # Plot combined ROC curves
        self.plot_multiple_roc_curves(all_fold_data)
        
        log.info("Cross-validation with ROC curves completed")

    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import roc_curve, auc

    def _get_fold_predictions(self):
        """
        Get predictions and true labels for the current fold's validation set.
        """
        self.model.eval()
        y_true = []
        y_prob = []
        
        with torch.no_grad():
            for batch_tup in self.val_dl:
                if len(batch_tup) == 4:
                    input_t, label_t, _, _ = batch_tup
                    labels = [1.0 if l_t.item() > self.config.threshold else 0.0 for l_t in label_t]
                    y_true.extend(labels)
                elif len(batch_tup) == 5:
                    input_t, _, has_ald_t, _, _ = batch_tup
                    y_true.extend(has_ald_t.float().cpu().numpy())
                
                input_g = input_t.to(self.device)
                probs = self.model(input_g).squeeze().cpu().numpy()
                
                # Handle single sample case
                if np.ndim(probs) == 0:
                    probs = np.array([probs])
                
                y_prob.extend(probs)
        
        return np.array(y_true), np.array(y_prob)
    
    def run_cross_validation_with_combined_roc(self, k=5):
        """
        Run k-fold cross-validation and plot combined ROC curves.
        """
        log.info(f"Starting {k}-fold cross-validation with combined ROC analysis")
        
        # Get all unique subjects
        all_subjects = list(set(self.input_df['anonymized_subject_id'].tolist()))
        
        # Stratify subjects based on whether they have ALD
        subjects_with_ald = []
        subjects_without_ald = []
        
        for subject in all_subjects:
            subject_rows = self.input_df[self.input_df['anonymized_subject_id'] == subject]
            max_loes = subject_rows['loes-score'].max()
            if max_loes > self.config.threshold:
                subjects_with_ald.append(subject)
            else:
                subjects_without_ald.append(subject)
        
        log.info(f"Total subjects: {len(all_subjects)} ({len(subjects_with_ald)} with ALD, {len(subjects_without_ald)} without ALD)")

        # Create stratified folds
        import random
        random.shuffle(subjects_with_ald)
        random.shuffle(subjects_without_ald)
    
        ald_folds = self._create_folds(subjects_with_ald, k)
        no_ald_folds = self._create_folds(subjects_without_ald, k)

        # Storage for ROC data from all folds - FIXED STRUCTURE
        fold_roc_data = {
            'original_fprs': [],    # Store original FPR arrays
            'original_tprs': [],    # Store original TPR arrays
            'interp_tprs': [],      # Store interpolated TPR arrays separately
            'aucs': [],
            'thresholds': [],
            'y_true_all': [],
            'y_prob_all': []
        }
        
        # Common FPR points for interpolation
        mean_fpr = np.linspace(0, 1, 100)
        
        # Metrics to track across folds
        fold_metrics = {
            'accuracy': [],
            'precision': [],
            'ppv': [],
            'recall': [],
            'f1': [],
            'auc': [],
            'sensitivity': [],
            'specificity': [],
            'pAUC': []
        }

        # Run training for each fold
        for fold in range(k):
            log.info(f"\n{'='*40}")
            log.info(f"FOLD {fold+1}/{k}")
            log.info(f"{'='*40}")
            
            # Create train/validation split for this fold
            val_subjects_ald = ald_folds[fold]
            val_subjects_no_ald = no_ald_folds[fold]
            
            train_subjects_ald = [s for i, fold_subjects in enumerate(ald_folds) for s in fold_subjects if i != fold]
            train_subjects_no_ald = [s for i, fold_subjects in enumerate(no_ald_folds) for s in fold_subjects if i != fold]
            
            self.train_subjects = train_subjects_ald + train_subjects_no_ald
            self.val_subjects = val_subjects_ald + val_subjects_no_ald
        
            log.info(f"Train set: {len(self.train_subjects)} subjects")
            log.info(f"Val set: {len(self.val_subjects)} subjects")
            
            # Set up dataloaders for this fold
            self.train_dl = self.data_handler.init_dl(self.folder, self.train_subjects)
            self.val_dl = self.data_handler.init_dl(self.folder, self.val_subjects, is_val_set=True)
            
            # Initialize a new model for this fold
            self._setup_model()
            self._setup_optimizer()
            self._setup_scheduler()
        
            # Train model for this fold
            fold_results = self._train_fold(fold)
            
            # Collect predictions for ROC analysis
            fold_y_true, fold_y_prob = self._get_fold_predictions()
            
            # Skip fold if we don't have both classes
            if len(np.unique(fold_y_true)) < 2:
                log.warning(f"Fold {fold+1} has only one class, skipping ROC calculation")
                continue
            
            # Calculate ROC curve for this fold
            fpr, tpr, thresholds = roc_curve(fold_y_true, fold_y_prob)
            fold_auc = auc(fpr, tpr)
            
            # Store fold data - FIXED: separate original and interpolated data
            fold_roc_data['original_fprs'].append(fpr)
            fold_roc_data['original_tprs'].append(tpr)
            fold_roc_data['aucs'].append(fold_auc)
            fold_roc_data['thresholds'].append(thresholds)
            fold_roc_data['y_true_all'].extend(fold_y_true)
            fold_roc_data['y_prob_all'].extend(fold_y_prob)
            
            # Interpolate TPR at common FPR points for averaging
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0  # Ensure it starts at (0,0)
            fold_roc_data['interp_tprs'].append(interp_tpr)  # Store separately
            
            # Track other metrics
            for metric in fold_metrics:
                if metric in fold_results:
                    fold_metrics[metric].append(fold_results[metric])
            
            log.info(f"Fold {fold+1} AUC: {fold_auc:.4f}")

        # Plot combined ROC curves
        self._plot_combined_roc_curves(fold_roc_data, mean_fpr, k)
        
        # Print cross-validation summary
        self._print_cv_summary(fold_metrics, fold_roc_data['aucs'])
        
        return fold_metrics, fold_roc_data


    def _plot_combined_roc_curves(self, fold_roc_data, mean_fpr, k):
        """
        Plot combined ROC curve showing mean performance, random classifier, and optimal threshold.
        """
        plt.figure(figsize=(10, 8))

        # Check if we have any valid folds
        if not fold_roc_data['original_fprs']:
            log.error("No valid folds with ROC data to plot")
            return
        
        # Calculate mean TPR across folds using interpolated data
        if fold_roc_data['interp_tprs']:
            mean_tpr = np.mean(fold_roc_data['interp_tprs'], axis=0)
            mean_tpr[-1] = 1.0  # Ensure it ends at (1,1)
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(fold_roc_data['aucs'])
            
            # Plot the mean ROC curve (blue line)
            plt.plot(mean_fpr, mean_tpr, color='blue', lw=3,
                    label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
            
            # Calculate and plot optimal threshold point on mean curve
            # Use Youden's J statistic to find optimal threshold
            youden_j = mean_tpr - mean_fpr
            optimal_idx = np.argmax(youden_j)
            optimal_fpr = mean_fpr[optimal_idx]
            optimal_tpr = mean_tpr[optimal_idx]
        
            # Calculate what threshold this corresponds to by using overall data
            if fold_roc_data['y_true_all'] and fold_roc_data['y_prob_all']:
                overall_fpr, overall_tpr, overall_thresholds = roc_curve(
                    fold_roc_data['y_true_all'], 
                    fold_roc_data['y_prob_all']
                )
                # Find closest point on overall curve to our optimal point
                distances = np.sqrt((overall_fpr - optimal_fpr)**2 + (overall_tpr - optimal_tpr)**2)
                closest_idx = np.argmin(distances)
                optimal_threshold = overall_thresholds[closest_idx]
                
                # Plot the optimal threshold point
                plt.plot(optimal_fpr, optimal_tpr, 'ro', markersize=10,
                        label=f'Optimal Threshold ({optimal_threshold:.3f})', zorder=5)
            else:
                # Fallback: just show the optimal point without threshold value
                plt.plot(optimal_fpr, optimal_tpr, 'ro', markersize=10,
                        label='Optimal Point (Youden\'s J)', zorder=5)

        # Plot diagonal (random classifier - dotted line)
        plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.8, label='Random Classifier')
        
        # Formatting
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title(f'CA-MEDS Performance', fontsize=16)
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)

        # Save the plot
        if hasattr(self.config, 'plot_location') and self.config.plot_location:
            log.info(f"Plot location config: {self.config.plot_location}")
            
            if os.path.isdir(self.config.plot_location):
                plot_path = os.path.join(self.config.plot_location, f"combined_roc_cv_{self.time_str}.png")
            else:
                base_path = os.path.splitext(self.config.plot_location)[0]
                plot_path = f"{base_path}_combined_roc_cv.png"
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            
            log.info(f"Attempting to save plot to: {plot_path}")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            
            # Verify the file was created
            if os.path.exists(plot_path):
                log.info(f"\u2713 Combined ROC curve successfully saved to {plot_path}")
            else:
                log.error(f"\u2717 Failed to save plot to {plot_path}")
        else:
            log.warning("No plot location configured, using fallback")
            plot_path = f"combined_roc_cv_{self.time_str}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            log.info(f"Combined ROC curve saved to {plot_path}")
        
        plt.close()  # Close the figure to free memory


    def _plot_overall_roc_curve(self, fold_roc_data):
        """
        Plot simplified overall ROC curve using all predictions from all folds combined.
        """
        plt.figure(figsize=(10, 8))
        
        # Calculate overall ROC using all predictions
        overall_fpr, overall_tpr, overall_thresholds = roc_curve(
            fold_roc_data['y_true_all'], 
            fold_roc_data['y_prob_all']
        )
        overall_auc = auc(overall_fpr, overall_tpr)
        
        # Plot overall ROC curve (clean, no threshold markers)
        plt.plot(overall_fpr, overall_tpr, color='red', lw=3,
                label=f'Overall ROC (AUC = {overall_auc:.3f})')

        # Plot diagonal
        plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.8, label='Random Classifier')
        
        # Formatting
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('CA-MEDS', fontsize=16)
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Save plot
        if hasattr(self.config, 'plot_location') and self.config.plot_location:
            base_path = os.path.splitext(self.config.plot_location)[0]
            plot_path = f"{base_path}_overall_roc_cv.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            log.info(f"Overall ROC curve saved to {plot_path}")
            
            # Also save as PDF
            pdf_path = f"{base_path}_overall_roc_cv.pdf"
            plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
            log.info(f"Overall ROC curve saved to {pdf_path}")
        else:
            # Fallback save location
            plot_path = f"overall_roc_cv_{self.time_str}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            log.info(f"Overall ROC curve saved to {plot_path}")
        
        plt.close()  # Close the figure to free memory
        
        return overall_auc

    def _print_cv_summary(self, fold_metrics, fold_aucs):
        """
        Print comprehensive cross-validation summary including ROC statistics.
        """
        log.info("\n" + "="*80)
        log.info("CROSS-VALIDATION RESULTS WITH ROC ANALYSIS")
        log.info("="*80)

        # Standard metrics
        for metric, values in fold_metrics.items():
            if values:
                mean_value = np.mean(values)
                std_value = np.std(values)
                log.info(f"{metric.capitalize()}: {mean_value:.4f} ± {std_value:.4f}")
        
        # ROC-specific statistics
        log.info(f"\nROC CURVE STATISTICS:")
        log.info(f"AUC per fold: {[f'{auc:.4f}' for auc in fold_aucs]}")
        log.info(f"Mean AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
        log.info(f"Min AUC: {np.min(fold_aucs):.4f}")
        log.info(f"Max AUC: {np.max(fold_aucs):.4f}")
        
        log.info("="*80)


if __name__ == "__main__":
    try:
        # Create app
        log_reg_app = LogisticRegressionApp()
        
        # Run the enhanced cross-validation with combined ROC (even in debug mode)
        if log_reg_app.config.DEBUG:
            log.info("Running in DEBUG mode - using combined ROC cross-validation")
            fold_metrics, fold_roc_data = log_reg_app.run_cross_validation_with_combined_roc(k=5)
        else:
            # Run the enhanced cross-validation with combined ROC
            fold_metrics, fold_roc_data = log_reg_app.run_cross_validation_with_combined_roc(k=5)
        
        log.info("Training and analysis completed successfully")
        
    except Exception as e:
        log.error(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
