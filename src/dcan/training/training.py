import argparse
import datetime
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR, OneCycleLR

from dcan.data_sets.dsets import LoesScoreDataset
from dcan.inference.make_predictions import add_predicted_values, compute_standardized_rmse, create_correlation_coefficient, create_scatter_plot, get_validation_info
from dcan.models.ResNet import get_resnet_model
from dcan.training.data_handler import DataHandler
from util.logconf import logging
from util.util import enumerateWithEstimate


# This script is a comprehensive deep learning pipeline for training a model to 
# predict Loes scores from MRI scans. It includes data loading, model selection,
# training/validation loops, logging, and model saving.


log = logging.getLogger(__name__)

# Used for computeBatchLoss and logMetrics to index into metrics_t/metrics_a
METRICS_LABEL_NDX = 0
METRICS_PRED_NDX = 1
METRICS_LOSS_NDX = 2
METRICS_SIZE = 3


# Refactored Configuration class to handle CLI arguments
class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--tb-prefix', default='loes_scoring', help="Tensorboard data prefix.")
        self.parser.add_argument('--csv-input-file', help="CSV data file.")
        self.parser.add_argument('--num-workers', default=8, type=int, help='Number of worker processes')
        self.parser.add_argument('--batch-size', default=32, type=int, help='Batch size for training')
        self.parser.add_argument('--epochs', default=1, type=int, help='Number of epochs to train')
        self.parser.add_argument('--file-path-column-index', type=int, help='Index of the file path in CSV file')
        self.parser.add_argument('--loes-score-column-index', type=int, help='Index of the Loes score in CSV file')
        self.parser.add_argument('--model-save-location', default=f'./model-{datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")}.pt')
        self.parser.add_argument('--plot-location', help='Location to save plot')
        self.parser.add_argument('--optimizer', default='Adam', help="Optimizer type.")
        self.parser.add_argument('comment', nargs='?', default='dcan', help="Comment for Tensorboard run")
        self.parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
        self.parser.add_argument('--gd', type=int, help="Use Gd-enhanced scans.")
        self.parser.add_argument('--use-train-validation-cols', action='store_true')
        self.parser.add_argument('-k', type=int, default=0, help='Index for 5-fold validation')
        self.parser.add_argument('--folder', help='Folder where MRIs are stored')
        self.parser.add_argument('--csv-output-file', help="CSV output file.")
        self.parser.add_argument('--use-weighted-loss', action='store_true')
        self.parser.add_argument(
            '--scheduler', default='plateau', 
            choices=['plateau', 'step', 'cosine', 'onecycle'], help='Learning rate scheduler')
        self.parser.add_argument(
            '--model', default='resnet', 
            choices=['resnet', 'alexnet'], help='Model architecture')

    def parse_args(self, sys_argv: list[str]) -> argparse.Namespace:
        return self.parser.parse_args(sys_argv)


# Model Handler Class to manage model operations
class ModelHandler:
    def __init__(self, model_name, use_cuda, device):
        self.model_name = model_name
        self.use_cuda = use_cuda
        self.device = device
        self.model = self._init_model()

    def _init_model(self):
        model = get_resnet_model()
        if torch.cuda.is_available():
            model.cuda()
        log.info("Using ResNet")

        if self.use_cuda and torch.cuda.device_count() > 1:
            log.info("Using CUDA with {} devices.".format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
            model = model.to(self.device)
        return model

    def save_model(self, save_location):
        if isinstance(self.model, torch.nn.DataParallel):
            self.model = self.model.module
        torch.save(self.model.state_dict(), save_location)

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

def count_items(input_list):
    counts = {}
    for item in input_list:
        if item in counts:
            counts[item] += 1
        else:
            counts[item] = 1
    return counts

def normalize_list(data):
    min_val = min(data)
    max_val = max(data)
    normalized_data = [(x - min_val) / (max_val - min_val) for x in data]
    return normalized_data

def normalize_dictionary(data):
    """
    Normalizes the values in a dictionary to a range between 0 and 1.

    Args:
        data (dict): A dictionary with numerical values.

    Returns:
        dict: A new dictionary with normalized values.
    """
    min_val = min(data.values())
    max_val = max(data.values())
    
    if max_val - min_val == 0:
      return {key: 0.0 for key in data}
    
    normalized_data = {
        key: (value - min_val) / (max_val - min_val)
        for key, value in data.items()
    }
    return normalized_data

# Training/Validation Loop Handler
class TrainingLoop:
    def __init__(self, model_handler, optimizer, device, df, config):
        self.model_handler = model_handler
        self.optimizer = optimizer
        self.device = device
        self.total_samples = 0
        self.df = df
        training_df = df[df['training'] == 1]
        loes_scores = list(training_df['loes-score'])
        item_counts = count_items(loes_scores)
        weighted_counts = {
            key: 1.0 / value
            for key, value in item_counts.items()
        }
        self.weights = weighted_counts
        self.config = config

    def train_epoch(self, epoch, train_dl):
        self.model_handler.model.train()
        trn_metrics_g = torch.zeros(METRICS_SIZE, len(train_dl.dataset), device=self.device)
        for batch_ndx, batch_tup in enumerateWithEstimate(train_dl, f"E{epoch} Training", start_ndx=train_dl.num_workers):
            self.optimizer.zero_grad()
            loss_var = self._compute_batch_loss(batch_ndx, batch_tup, train_dl.batch_size, trn_metrics_g)
            loss_var.backward()
            self.optimizer.step()
        self.total_samples += len(train_dl.dataset)
        return trn_metrics_g.to('cpu')

    def validate_epoch(self, epoch, val_dl):
        with torch.no_grad():
            self.model_handler.model.eval()
            val_metrics_g = torch.zeros(METRICS_SIZE, len(val_dl.dataset), device=self.device)
            for batch_ndx, batch_tup in enumerateWithEstimate(val_dl, f"E{epoch} Validation", start_ndx=val_dl.num_workers):
                self._compute_batch_loss(batch_ndx, batch_tup, val_dl.batch_size, val_metrics_g)
        return val_metrics_g.to('cpu')
    
    def weighted_mse_loss(self, predictions, targets):
        """
        Calculate weighted MSE loss for regression where weights are determined by
        the frequency of each target value in the training set.
        
        Args:
            predictions (torch.Tensor): Model predictions, shape [batch_size]
            targets (torch.Tensor): Ground truth values, shape [batch_size]
            
        Returns:
            torch.Tensor: Weighted MSE loss for each sample in the batch
        """
        # Convert targets to CPU for dictionary lookup if they're on GPU
        targets_cpu = targets.detach().cpu().numpy()
        
        # Create a tensor to store weights for each sample in the batch
        weights = torch.ones_like(predictions)
        
        # Assign weights based on target values
        for i, target in enumerate(targets_cpu):
            # Handle potential floating point issues by rounding
            target_key = round(float(target), 1)  # Adjust rounding precision as needed
            
            # Get weight from dictionary, default to 1.0 if not found
            if target_key in self.weights:
                weights[i] = torch.tensor(self.weights[target_key], 
                                        device=predictions.device)
            else:
                # For unseen values, use the mean weight or a default
                mean_weight = sum(self.weights.values()) / len(self.weights)
                weights[i] = torch.tensor(mean_weight, device=predictions.device)
        
        # Calculate weighted squared error for each sample
        squared_errors = (predictions - targets) ** 2
        weighted_squared_errors = weights * squared_errors
        
        return weighted_squared_errors
    
    def _compute_batch_loss(self, batch_ndx, batch_tup, batch_size, metrics_g):
        input_t, label_t, _, _ = batch_tup
        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        outputs_g = self.model_handler.model(input_g)

        # If outputs_g is a list or tuple, take the first element
        if isinstance(outputs_g, (list, tuple)):
            outputs_g = outputs_g[0]  

        outputs_g = outputs_g.squeeze(dim=-1)  # Remove extra dimension if needed

        label_g = label_g.view(-1)  # Ensures shape is [batch_size]

        log.debug(f"outputs_g shape: {outputs_g.shape}")  # Should be [batch_size]
        log.debug(f"label_g shape: {label_g.shape}")  # Should be [batch_size]

        # When using Regressor for the model, it is important that we use nn.MSELoss for regression.
        if self.config.use_weighted_loss:
            loss_g = self.weighted_mse_loss(outputs_g, label_g)
            loss_mean = loss_g.mean()  # Get mean for backpropagation
        else:
            loss_func = nn.MSELoss(reduction='none')
            loss_g = loss_func(outputs_g, label_g)
            loss_mean = loss_g.mean()

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)
        
        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = \
            label_g.detach()
        metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = \
            outputs_g.detach()
        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = \
            loss_mean.detach()

        return loss_mean


# TensorBoard Logger
class TensorBoardLogger:
    def __init__(self, tb_prefix, time_str, comment):
        self.log_dir = os.path.join('runs', tb_prefix, time_str)
        self.trn_writer = SummaryWriter(log_dir=self.log_dir + f'-trn_cls-{comment}')
        self.val_writer = SummaryWriter(log_dir=self.log_dir + f'-val_cls-{comment}')

    def log_metrics(self, mode_str, epoch, metrics, sample_count):
        writer = getattr(self, f'{mode_str}_writer')
        writer.add_scalar(f'loss/all', metrics[METRICS_LOSS_NDX].mean(), sample_count)

    def close(self):
        self.trn_writer.close()
        self.val_writer.close()

def get_folder_name(file_path):
  """
  Extracts the folder name from a given file path.

  Args:
    file_path: The path to the file.

  Returns:
    The name of the folder containing the file, or None if an error occurs.
  """
  try:
    folder_path = os.path.dirname(file_path)
    folder_name = os.path.basename(folder_path)
    return folder_name
  except Exception as e:
    print(f"An error occurred: {e}")
    return None

# Main Application Class
class LoesScoringTrainingApp:        
    def __init__(self, sys_argv=None):
        self.config = Config().parse_args(sys_argv)
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_handler = ModelHandler(self.config.model, self.use_cuda, self.device)
        self.optimizer = self._init_optimizer()

        self.input_df = pd.read_csv(self.config.csv_input_file)
        self.output_df = self.input_df.copy()

        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.tb_logger = TensorBoardLogger(self.config.tb_prefix, self.time_str, self.config.comment)

        self.data_handler = DataHandler(self.input_df, self.output_df, self.use_cuda, self.config.batch_size, self.config.num_workers)
        self.folder = self.config.folder

    def _init_optimizer(self):
        optimizer_type = self.config.optimizer.lower()
        optimizer_cls = Adam if optimizer_type == 'adam' else SGD
        return optimizer_cls(self.model_handler.model.parameters(), lr=self.config.lr)
        
    def _init_scheduler(self, train_dl):
        # '--scheduler', default='plateau', 
        #    choices=['plateau', 'step', 'cosine', 'onecycle']
        if self.config.scheduler == 'step':
            scheduler = StepLR(self.optimizer, step_size=30, gamma=0.1)
        elif self.config.scheduler == 'cosine':
            scheduler = CosineAnnealingLR(self.optimizer, T_max=100, eta_min=0)
        elif self.config.scheduler == 'onecycle':
            scheduler = OneCycleLR(
                self.optimizer, 
                max_lr=0.01,
                total_steps=len(train_dl) * self.config.epochs,
                pct_start=0.3
            )
        else:
            scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10)
        
        return scheduler

    def main(self):
        log.info("Starting training...")
        self.output_df["prediction"] = np.nan

        if self.config.gd == 0:
            self.input_df = self.input_df[~self.input_df['scan'].str.contains('Gd')]

        if self.config.use_train_validation_cols:
            training_rows = self.input_df.loc[self.input_df['training'] == 1]
            train_subjects = list(training_rows['anonymized_subject_id'])
            validation_rows = self.input_df.loc[self.input_df['validation'] == 1]
            val_subjects = list(validation_rows['anonymized_subject_id'])
        else:
            train_subjects, val_subjects = self.split_train_validation()

        self.train_dl = self.data_handler.init_dl(self.folder, train_subjects)
        val_dl = self.data_handler.init_dl(self.folder, val_subjects, is_val_set=True)
        
        # Add scheduler initialization
        self.scheduler = self._init_scheduler(self.train_dl)
        
        loop_handler = TrainingLoop(self.model_handler, self.optimizer, self.device, self.input_df, self.config)
        
        best_val_loss = float('inf')
        
        for epoch in range(1, self.config.epochs + 1):
            log.info(f"Epoch {epoch}/{self.config.epochs}")

            trn_metrics = loop_handler.train_epoch(epoch, self.train_dl)
            val_metrics = loop_handler.validate_epoch(epoch, val_dl)

            self.tb_logger.log_metrics('trn', epoch, trn_metrics, loop_handler.total_samples)
            self.tb_logger.log_metrics('val', epoch, val_metrics, loop_handler.total_samples)
            
            # Calculate validation loss
            val_loss = val_metrics[METRICS_LOSS_NDX].mean().item()
            
            # Step the scheduler based on validation loss
            self.scheduler.step(val_loss)
            
            # Track best model (optional)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                log.info(f"New best validation loss: {best_val_loss}")
                # Save the best model here
                self.model_handler.save_model(self.config.model_save_location)
            
            # Log current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            log.info(f"Current learning rate: {current_lr}")

        input_csv_location = self.config.csv_input_file
        subjects, sessions, actual_scores, predict_vals = \
            get_validation_info(self.config.model, self.config.model_save_location, input_csv_location)
        output_csv_location = self.config.csv_output_file
        output_df = add_predicted_values(subjects, sessions, predict_vals, input_csv_location)
        output_csv_folder_name = get_folder_name(output_csv_location)
        if not os.path.exists(output_csv_folder_name):
            os.makedirs(output_csv_folder_name)
        output_df.to_csv(output_csv_location, index=False)
        standardized_rmse = \
            compute_standardized_rmse(actual_scores, predict_vals)
        log.info(f'standardized_rmse: {standardized_rmse}')
        create_scatter_plot(actual_scores, predict_vals, self.config.plot_location)
        correlation_coefficient = create_correlation_coefficient(actual_scores, predict_vals)
        log.info(f'correlation_coefficient: {correlation_coefficient}')

        _, p_value = stats.pearsonr(actual_scores, predict_vals)
        log.info(f"Pearson correlation p-value: {p_value}")

        _, p_value = stats.spearmanr(actual_scores, predict_vals)
        log.info(f"Spearman correlation p-value: {p_value}")

    def get_files_with_wildcard(self, directory, pattern):
        """
        Gets a list of files in a directory that match a wildcard pattern.

        Args:
            directory: The directory to search in.
            pattern: The wildcard pattern to match (e.g., "*.txt", "image*").

        Returns:
            A list of file paths that match the pattern.
        """
        search_path = os.path.join(directory, pattern)
        files = glob.glob(search_path)
        return files

    def split_train_validation(self):
        training_rows = self.input_df.loc[self.input_df['training'] == 1]
        validation_rows = self.input_df.loc[self.input_df['validation'] == 1]
        validation_users = list(set(validation_rows['anonymized_subject_id'].to_list()))
        training_users = list(set(training_rows['anonymized_subject_id'].to_list()))
        
        return training_users, validation_users

def main():
    loesScoringTrainingApp = LoesScoringTrainingApp(sys_argv=sys.argv)
    loesScoringTrainingApp.main()


if __name__ == "__main__":
    main()
