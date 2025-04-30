import argparse
import copy
import datetime
import os
import random
import sys
from typing import List
import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR, OneCycleLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm

from dcan.data_sets.dsets import CandidateInfoTuple, LoesScoreDataset
import torch
from dcan.training.augmented_loes_score_dataset import AugmentedLoesScoreDataset
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

        return DataLoader(dataset, batch_size=batch_size, num_workers=self.num_workers, pin_memory=self.use_cuda)

# Metrics indices
METRICS_LABEL_NDX = 0
METRICS_PRED_NDX = 1
METRICS_PROB_NDX = 2
METRICS_LOSS_NDX = 3
METRICS_SIZE = 4

def append_candidate(folder, candidate_info_list, row):
    subject_str = row['anonymized_subject_id']
    session_str = row['anonymized_session_id']
    file_name = f"{subject_str}_{session_str}_space-MNI_brain_mprage_RAVEL.nii.gz"
    file_path = os.path.join(folder, file_name)
    
    # Check if file exists
    if not os.path.exists(file_path):
        log.warning(f"MRI file not found: {file_path}")
        return False
        
    try:
        loes_score_float = float(row['loes-score'])
    except (ValueError, TypeError):
        log.warning(f"Invalid LOES score for subject {subject_str}, session {session_str}: {row['loes-score']}")
        return False
        
    candidate_info_list.append(CandidateInfoTuple(
        loes_score_float,
        file_path,
        subject_str,
        session_str
    ))
    return True

def get_candidate_info_list(folder, df, candidates: List[str]):
    candidate_info_list = []
    df = df.reset_index()  # make sure indexes pair with number of rows

    for _, row in df.iterrows():
        candidate = row['anonymized_subject_id']
        if candidate in candidates:
            append_candidate(folder, candidate_info_list, row)

    candidate_info_list.sort(reverse=True)

    return candidate_info_list


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
        self.parser.add_argument('--class-weights', action='store_true', help='Use class weights for imbalanced data')
        self.parser.add_argument('--gd', action='store_true', help='Use gadolinium-enhanced MRIs.')
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


# Custom dataset for logistic regression
class LogisticRegressionDataset(Dataset):
    def __init__(self,
                 folder,
                 subjects: List[str], df, output_df,
                 is_val_set_bool=None,
                 subject=None,
                 sortby_str='random'
                 ):
        self.df = df
        self.is_val_set_bool = is_val_set_bool
        self.candidateInfo_list = copy.copy(get_candidate_info_list(folder, df, subjects))

        if subject:
            self.candidateInfo_list = [
                x for x in self.candidateInfo_list if x.subject_str == subject
            ]

        if sortby_str == 'random':
            random.shuffle(self.candidateInfo_list)
        elif sortby_str == 'loes_score':
            pass
        else:
            raise Exception("Unknown sort: " + repr(sortby_str))

        log.info("{!r}: {} {} samples".format(
            self,
            len(self.candidateInfo_list),
            "validation" if is_val_set_bool else "training",
        ))
        if output_df is not None:
            for candidate_info in self.candidateInfo_list:
                row_location = (df["anonymized_subject_id"] == candidate_info.subject) & (df["anonymized_session_id"] == candidate_info.session_str)
                output_df.loc[row_location, 'training'] = 0 if is_val_set_bool else 1
                output_df.loc[row_location, 'validation'] = 1 if is_val_set_bool else 0
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        log.debug("Calling __getitem__(self, idx)")
        return self.features[idx], self.target[idx]
    
    def get_feature_columns(self):
        return self.df.columns.tolist()


# Simple logistic regression model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        logits = self.linear(x)
        return torch.sigmoid(logits)


# Training and validation loop handler
class LogisticRegressionTrainer:
    def __init__(self, model, optimizer, device, class_weights=None, threshold=0.5):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.total_samples = 0
        self.threshold = threshold
        
        # Set up loss function
        self.loss_fn = nn.BCELoss(reduction='none')
    
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


# Main Application Class
# Update the initialization sequence in your LogisticRegressionApp class
class LogisticRegressionApp:
    def __init__(self, sys_argv=None):
        self.config = Config().parse_args(sys_argv)
        self.class_weights = self.config.class_weights
        if not self.config.DEBUG:
            self.use_cuda = torch.cuda.is_available()
            self.device = torch.device("cuda" if self.use_cuda else "cpu")
        else:
            self.use_cuda = False
            self.device = "cpu"
            
        # Load the data (only once)
        self._load_data()

        # Check data integrity before proceeding
        self.check_data_integrity()
        
        # Setup model, datasets, and training components IN THE CORRECT ORDER
        self.folder = self.config.folder
        
        # 1. First, set up the model (needs to be done before optimizer)
        self._setup_model()
        
        # 2. Then initialize the data handler and datasets
        self.data_handler = DataHandler(self.input_df, self.output_df, self.use_cuda, self.config.batch_size, self.config.num_workers)
        self.data_handler = DataHandler(
            self.input_df, self.output_df, self.use_cuda, 
            self.config.batch_size, self.config.num_workers
        )
        self.data_handler.augment_minority = self.config.augment_minority

        # Print data summary
        log.info(f"Dataset shape: {self.input_df.shape}")
        log.info(f"Features: {self.config.features}")
        log.info(f"Target: {self.config.target}")
        
        # Setup augmentation (add this line)
        self._setup_augmentation()
        
        # Setup model, datasets, and training components
        self.folder = self.config.folder

        self._setup_datasets()
        self._setup_dataloaders()
        
        # 3. Only after model is set up, initialize optimizer and scheduler
        print(list(self.model.parameters()))
        self._setup_optimizer()
        self._setup_scheduler()
        
        # Setup logging
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.tb_logger = TensorBoardLogger(self.config.tb_prefix, self.time_str, self.config.comment)

    def check_data_integrity(self):
        """
        Performs various checks to ensure data integrity before training.
        Raises appropriate exceptions if issues are found.
        """
        log.info("Checking data integrity...")
        
        # Check class balance
        target_values = self.input_df[self.config.target].values
        class_counts = np.bincount(target_values.astype(int))
        
        if len(class_counts) <= 1:
            raise ValueError(f"Only one class found in target column '{self.config.target}'. Cannot train a classifier.")
        
        minority_class_count = min(class_counts)
        majority_class_count = max(class_counts)
        imbalance_ratio = majority_class_count / minority_class_count if minority_class_count > 0 else float('inf')
        
        if imbalance_ratio > 10:
            log.warning(f"Severe class imbalance detected: ratio of {imbalance_ratio:.2f}. "
                        f"Consider using --class-weights or --augment-minority.")
        
        # Check feature distributions
        for feature in self.config.features:
            feature_values = self.input_df[feature].values
            if len(np.unique(feature_values)) == 1:
                log.warning(f"Feature '{feature}' has only one unique value. Consider removing it.")
            
            # Check for outliers using IQR
            if np.issubdtype(feature_values.dtype, np.number):
                q1 = np.percentile(feature_values, 25)
                q3 = np.percentile(feature_values, 75)
                iqr = q3 - q1
                outlier_mask = (feature_values < q1 - 1.5 * iqr) | (feature_values > q3 + 1.5 * iqr)
                outlier_count = np.sum(outlier_mask)
                
                if outlier_count > 0:
                    outlier_pct = outlier_count / len(feature_values) * 100
                    if outlier_pct > 5:
                        log.warning(f"Feature '{feature}' has {outlier_pct:.1f}% outliers. "
                                    f"Consider normalization or outlier handling.")
    
        # Check for MRI file existence if applicable
        if self.config.folder:
            if not os.path.exists(self.config.folder):
                raise FileNotFoundError(f"MRI folder not found: {self.config.folder}")
                
            # Check a sample of MRI files
            sample_subjects = random.sample(list(self.input_df['anonymized_subject_id'].unique()), 
                                            min(5, len(self.input_df['anonymized_subject_id'].unique())))
                                        
            missing_files = 0
            for subject in sample_subjects:
                sample_rows = self.input_df[self.input_df['anonymized_subject_id'] == subject].iloc[:1]
                for _, row in sample_rows.iterrows():
                    subject_str = row['anonymized_subject_id']
                    session_str = row['anonymized_session_id']
                    file_name = f"{subject_str}_{session_str}_space-MNI_brain_mprage_RAVEL.nii.gz"
                    file_path = os.path.join(self.config.folder, file_name)
                    
                    if not os.path.exists(file_path):
                        missing_files += 1
                        
            if missing_files > 0:
                log.warning(f"{missing_files} out of {len(sample_subjects)} sampled MRI files are missing. "
                            f"This might indicate data availability issues.")
        
        log.info("Data integrity check completed.")

    def _setup_model(self):
        """
        Set up the model based on configuration.
        This method handles model initialization with proper error checking.
        """
        model_type = self.config.model_type
        debug_mode = self.config.DEBUG
        
        log.info(f"Setting up {model_type} model")
        
        try:
            # Create model
            if model_type == 'simple':
                self.model = SimpleMRIModel(debug=debug_mode)
            elif model_type == 'conv':
                self.model = get_mri_logistic_regression_model(model_type='conv', debug=debug_mode)
            elif model_type in ['resnet3d', 'dense3d', 'efficientnet3d']:
                # Import only if needed
                from dcan.models.advanced_mri_models import get_advanced_mri_model
                self.model = get_advanced_mri_model(model_type=model_type, debug=debug_mode)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
            # Move model to device
            self.model = self.model.to(self.device)
            log.info(f"Model moved to {self.device}")
            
            # Log model parameter count
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            log.info(f"Model has {total_params:,} total parameters ({trainable_params:,} trainable)")
        
            # Ensure model is in training mode
            self.model.train()
            
        except ImportError as e:
            log.error(f"Failed to import necessary modules for model type '{model_type}': {e}")
            raise
        except Exception as e:
            log.error(f"Error creating model: {e}")
            raise
            
        log.info(f"Model initialization complete: {model_type}")

    def _load_data(self):
        log.info(f"Loading data from {self.config.csv_input_file}")
        try:
            if not os.path.exists(self.config.csv_input_file):
                raise FileNotFoundError(f"Input file not found: {self.config.csv_input_file}")
                
            self.input_df = pd.read_csv(self.config.csv_input_file)
            
            # Validate that required columns exist
            missing_features = [f for f in self.config.features if f not in self.input_df.columns]
            if missing_features:
                raise ValueError(f"Missing required feature columns: {missing_features}")
                
            if self.config.target not in self.input_df.columns:
                raise ValueError(f"Target column '{self.config.target}' not found in input file")
                
            self.output_df = self.input_df.copy()
            self.output_df["prediction"] = np.nan
            
            # Check for missing values
            missing_values = self.input_df[self.config.features + [self.config.target]].isnull().sum()
            if missing_values.sum() > 0:
                for col, count in missing_values.items():
                    if count > 0:
                        log.warning(f"Found {count} missing values in column '{col}'")
            
            if hasattr(self.config, 'gd') and self.config.gd:
                before_count = len(self.input_df)
                self.input_df = self.input_df[~self.input_df['scan'].str.contains('Gd')]
                after_count = len(self.input_df)
                log.info(f"Removed {before_count - after_count} Gd scans from dataset")
            
            # Print data summary
            log.info(f"Dataset shape: {self.input_df.shape}")
            log.info(f"Features: {self.config.features}")
            log.info(f"Target: {self.config.target}")
        
        except pd.errors.EmptyDataError:
            log.error(f"CSV file is empty: {self.config.csv_input_file}")
            raise
        except pd.errors.ParserError:
            log.error(f"Error parsing CSV file: {self.config.csv_input_file}")
            raise
        except Exception as e:
            log.error(f"Error loading data: {e}")
            raise

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

    def _setup_datasets(self):
        """
        Prepare train/validation datasets by extracting subjects from the dataframes.
        This method doesn't actually create the datasets but prepares the lists
        of subjects that will be used in _setup_dataloaders.
        """
        log.info("Setting up datasets...")
        
        # Extract training and validation subjects
        if self.config.use_train_validation_cols:
            # Use existing training/validation flags from the DataFrame
            training_rows = self.input_df[self.input_df['training'] == 1]
            self.train_subjects = list(set(training_rows['anonymized_subject_id'].tolist()))
            
            validation_rows = self.input_df[self.input_df['validation'] == 1]
            self.val_subjects = list(set(validation_rows['anonymized_subject_id'].tolist()))
        else:
            # Split based on config split ratio
            self.train_subjects, self.val_subjects = self.split_train_validation()
        
        log.info(f"Train set: {len(self.train_subjects)} subjects")
        log.info(f"Val set: {len(self.val_subjects)} subjects")

    # If you're moving away from the dataset-oriented approach and using DataHandler directly,
    # update your _setup_dataloaders method to match:

    def _setup_dataloaders(self):
        """
        Create DataLoader objects for training and validation using the DataHandler.
        """
        log.info("Setting up dataloaders...")
        
        # Use DataHandler to create the data loaders
        self.train_dl = self.data_handler.init_dl(self.folder, self.train_subjects)
        self.val_dl = self.data_handler.init_dl(self.folder, self.val_subjects, is_val_set=True)
        
        log.info(f"Train dataloader: {len(self.train_dl)} batches")
        log.info(f"Val dataloader: {len(self.val_dl)} batches")

    def split_train_validation(self):
        training_rows = self.input_df.loc[self.input_df['training'] == 1]
        validation_rows = self.input_df.loc[self.input_df['validation'] == 1]
        validation_users = list(set(validation_rows['anonymized_subject_id'].to_list()))
        training_users = list(set(training_rows['anonymized_subject_id'].to_list()))
        
        return training_users, validation_users
    
    def _setup_scheduler(self):
        """
        Set up the learning rate scheduler based on the configuration.
        This method should be called after the optimizer has been initialized.
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

    def _setup_augmentation(self):
        """Identify minority class samples for augmentation"""
        log.info("Setting up augmentation for minority class...")
        
        # Get all samples
        all_subjects = set(self.input_df['anonymized_subject_id'].tolist())
        
        # Identify subjects without ALD (minority class)
        self.non_ald_subjects = []
        for subject in all_subjects:
            subject_rows = self.input_df[self.input_df['anonymized_subject_id'] == subject]
            max_loes = subject_rows['loes-score'].max()
            if max_loes <= self.config.threshold:
                self.non_ald_subjects.append(subject)
        
        log.info(f"Found {len(self.non_ald_subjects)} subjects in minority class (non-ALD)")

    def run_cross_validation(self, k=5):
        """
        Run k-fold cross-validation on the dataset.
        
        Args:
            k (int): Number of folds for cross-validation
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
            'specificity': []
        }
        
        # Save path for the best model across all folds
        best_model_path = self.config.model_save_location
        best_val_f1 = 0.0
        
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
            
            # Initialize dataloaders
            self._setup_dataloaders()
            
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
            if fold_results.get('f1', 0) > best_val_f1:
                best_val_f1 = fold_results.get('f1', 0)
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

    def _train_fold(self, fold_idx):
        """
        Train and evaluate model for a single fold.
        
        Args:
            fold_idx (int): Index of the current fold
            
        Returns:
            dict: Dictionary of validation metrics for this fold
        """
        # Initialize the trainer
        trainer = LogisticRegressionTrainer(
            self.model, 
            self.optimizer, 
            self.device, 
            self.class_weights if hasattr(self, 'class_weights') else None,
            threshold=self.config.threshold
        )
    
        # Track best model
        best_val_f1 = 0.0
        best_epoch = 0
        fold_metrics = {}
        
        # Train for specified number of epochs
        for epoch in range(1, self.config.epochs + 1):
            log.info(f"Fold {fold_idx+1} - Epoch {epoch}/{self.config.epochs}")
            
            # Training phase
            trn_metrics = trainer.train_epoch(epoch, self.train_dl)
            trn_loss = trn_metrics[METRICS_LOSS_NDX].mean().item()
            
            # Validation phase
            val_metrics = trainer.validate_epoch(epoch, self.val_dl)
            val_loss = val_metrics[METRICS_LOSS_NDX].mean().item()
            
            # Calculate metrics
            y_true = val_metrics[METRICS_LABEL_NDX].numpy()
            y_pred = val_metrics[METRICS_PRED_NDX].numpy()
            y_prob = val_metrics[METRICS_PROB_NDX].numpy()
            
            try:
                # Calculate classification metrics
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, zero_division=0)
                ppv = precision  # PPV is identical to precision
                recall = sensitivity = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
            
                # Compute AUC if we have both classes
                auc = 0.5
                if len(np.unique(y_true)) > 1:
                    auc = roc_auc_score(y_true, y_prob)
                
                # Compute confusion matrix
                cm = confusion_matrix(y_true, y_pred)
                
                # Calculate specificity
                specificity = 0.0
                if cm.shape == (2, 2) and cm[0][0] + cm[0][1] > 0:
                    specificity = cm[0][0] / (cm[0][0] + cm[0][1])
                
                # Update learning rate scheduler
                if self.config.scheduler == 'plateau':
                    self.scheduler.step(val_loss)
                elif self.config.scheduler != 'onecycle':
                    self.scheduler.step()
                
                # Track best F1 score
                if f1 > best_val_f1:
                    best_val_f1 = f1
                    best_epoch = epoch
                    
                    # Save metrics for best epoch
                    fold_metrics = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'ppv': ppv,
                        'recall': recall,
                        'f1': f1,
                        'auc': auc,
                        'sensitivity': sensitivity,
                        'specificity': specificity,
                        'loss': val_loss
                    }
                    
                    # Log best model info
                    log.info(f"New best F1: {f1:.4f} (accuracy: {accuracy:.4f}, AUC: {auc:.4f})")
                
                # Log epoch summary
                log.info(f"Fold {fold_idx+1} - Epoch {epoch} - "
                        f"Loss: {val_loss:.4f}, "
                        f"Accuracy: {accuracy:.4f}, "
                        f"F1: {f1:.4f}, "
                        f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
                
            except Exception as e:
                log.error(f"Error computing metrics: {e}")
        
        log.info(f"Fold {fold_idx+1} completed. Best F1: {best_val_f1:.4f} (epoch {best_epoch})")
        return fold_metrics

    def generate_summary_statistics(self):
        """
        Generate and display comprehensive summary statistics after training.
        This includes model performance, training history, dataset information,
        and other relevant metrics.
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
        # Use the model to predict on validation data
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
                    label = [1.0 if l_t.item() > 0 + self.config.threshold else 0.0 for l_t in label_t]
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
                auc = roc_auc_score(val_labels, val_probs)
                log.info(f"AUC: {auc:.4f}")
            
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

    def train(self):
        """
        Execute the training loop over multiple epochs.
        This method handles training, validation, logging, and model checkpointing.
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
                self.class_weights if hasattr(self, 'class_weights') else None,
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
                    self.scheduler.step(val_loss)  # Pass validation loss to ReduceLROnPlateau
                elif self.config.scheduler != 'onecycle':  # Step and Cosine schedulers step each epoch
                    self.scheduler.step()
                # Note: OneCycleLR steps after each batch, not each epoch
                
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
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                log.error("CUDA out of memory error. Try reducing batch size or using a smaller model.")
            elif "Expected object of device" in str(e):
                log.error("Device mismatch error. Check that all tensors are on the same device.")
            else:
                log.error(f"Runtime error during training: {e}")
            raise
        except Exception as e:
            log.error(f"Error during training: {e}")
            raise

    def save_model(self, path):
        """Save the trained model to disk"""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save(self.model.state_dict(), path)
        log.info(f"Model saved to {path}")

# Fallback simple model in case mri_logistic_regression.py isn't available
class SimpleMRIModel(nn.Module):
    def __init__(self, debug=False):
        super(SimpleMRIModel, self).__init__()
        self.debug = debug
        self.initialized = False
        
    def _initialize(self, input_size):
        # Create a simple feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU()
        )
        self.classifier = nn.Linear(32, 1)
        self.initialized = True
        
    def forward(self, x):
        log.debug(f"Input shape: {x.shape}")
        
        # Reshape input to 2D (batch_size, features)
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        
        log.debug(f"Flattened shape: {x_flat.shape}")
        
        # Initialize on first forward pass
        if not self.initialized:
            self._initialize(x_flat.size(1))
            log.debug(f"Model initialized with input size: {x_flat.size(1)}")
        
        # Apply feature extraction
        features = self.feature_extractor(x_flat)
        
        log.debug(f"Features shape: {features.shape}")
        
        # Apply final classification
        logits = self.classifier(features)
        
        return torch.sigmoid(logits)    

def main():
    log_reg_app = LogisticRegressionApp()
    
    # Use cross-validation instead of single train/validation
    log_reg_app.run_cross_validation(k=5)  # 5-fold cross-validation

if __name__ == "__main__":
    main()