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

from dcan.data_sets.dsets import get_candidate_info_list

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)

# Metrics indices
METRICS_LABEL_NDX = 0
METRICS_PRED_NDX = 1
METRICS_PROB_NDX = 2
METRICS_LOSS_NDX = 3
METRICS_SIZE = 4

# Configuration class to handle CLI arguments
class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--DEBUG', action='store_false')
        # Data parameters
        self.parser.add_argument('--csv-input-file', required=True, help="CSV data file")
        self.parser.add_argument('--csv-output-file', help="CSV output file for predictions")
        self.parser.add_argument('--features', required=True, nargs='+', help="Feature column names")
        self.parser.add_argument('--target', required=True, help="Target column name")
        self.parser.add_argument('--folder', help='Folder where MRIs are stored')
        
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
        self.parser.add_argument('--gd', action='store_false', help='Use gadolinium-enhanced MRIs.')
        
        # Output and tracking
        self.parser.add_argument('--tb-prefix', default='logistic_regression', help="Tensorboard data prefix")
        self.parser.add_argument('--model-save-location', help='Location to save model')
        self.parser.add_argument('--plot-location', help='Location to save plots')
        self.parser.add_argument('--comment', default='', help="Comment for Tensorboard run")
        self.parser.add_argument('--normalize-features', action='store_true', help='Normalize input features')
        
        # Evaluation
        self.parser.add_argument('--threshold', default=0.5, type=float, help='Classification threshold')

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
        if class_weights is not None:
            # Convert weights to tensor and move to device
            weight_tensor = torch.tensor([class_weights[0], class_weights[1]], dtype=torch.float32).to(device)
            self.loss_fn = nn.BCELoss(weight=weight_tensor, reduction='none')
        else:
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
        input_t, label_t = batch_tup
        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)
        
        prob_g = self.model(input_g)
        prob_g = prob_g.squeeze(dim=-1)  # [batch_size]
        pred_g = (prob_g >= self.threshold).float()
        
        # Ensure label tensor has the right shape
        label_g = label_g.view(-1)  # [batch_size]
        
        # Compute loss
        loss_g = self.loss_fn(prob_g, label_g)
        loss_mean = loss_g.mean()
        
        # Store metrics
        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)
        
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
class LogisticRegressionApp:
    def __init__(self, sys_argv=None):
        self.config = Config().parse_args(sys_argv)
        if not self.config.DEBUG:
            self.use_cuda = torch.cuda.is_available()
            self.device = torch.device("cuda" if self.use_cuda else "cpu")
        else:
            self.use_cuda = False
            self.device = "cpu"
        
        # Load the data
        log.info(f"Loading data from {self.config.csv_input_file}")
        self.input_df = pd.read_csv(self.config.csv_input_file)
        self.output_df = self.input_df.copy()
        self.output_df["prediction"] = np.nan

        if self.config.gd:
            self.input_df = self.input_df[~self.input_df['scan'].str.contains('Gd')]

        # Print data summary
        log.info(f"Dataset shape: {self.input_df.shape}")
        log.info(f"Features: {self.config.features}")
        log.info(f"Target: {self.config.target}")
        
        # Check for missing values
        missing_values = self.input_df[self.config.features + [self.config.target]].isnull().sum().sum()
        if missing_values > 0:
            log.warning(f"Found {missing_values} missing values in the dataset")
        
        # Setup model and training components
        self._setup_model()
        self._setup_datasets()
        self._setup_dataloaders()
        self._setup_optimizer()
        self._setup_scheduler()
        
        # Setup logging
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.tb_logger = TensorBoardLogger(self.config.tb_prefix, self.time_str, self.config.comment)
    
    def _setup_model(self):
        input_dim = len(self.config.features)
        self.model = LogisticRegressionModel(input_dim=input_dim)
        
        if self.use_cuda:
            self.model = self.model.to(self.device)
            log.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        else:
            log.info("Using CPU")
    
    def _setup_datasets(self):
        # Prepare train/val split
        train_df = self.input_df.loc[self.input_df['training'] == 1]
        train_subjects = train_df['anonymized_subject_id'].tolist()
        val_df = self.input_df.loc[self.input_df['validation'] == 1]
        
        log.info(f"Train set: {len(train_df)} samples, Val set: {len(val_df)} samples")
        
        # Create datasets
        self.train_dataset = LogisticRegressionDataset(
            self.config.folder,
            train_subjects, 
            train_df, 
            train_df.copy(),
            is_val_set_bool=False
        )
        
        val_subjects = val_df['anonymized_subject_id'].tolist()
        self.val_dataset = LogisticRegressionDataset(
            self.config.folder,
            val_subjects, 
            val_df, 
            val_df.copy(),
            is_val_set_bool=True
        )
        
        # Calculate class weights if needed
        if self.config.class_weights:
            target_counts = train_df[self.config.target].value_counts()
            total_samples = len(train_df)
            
            # Class weights are inversely proportional to class frequencies
            self.class_weights = {
                0: total_samples / (2 * (total_samples - target_counts.get(1, 0))),
                1: total_samples / (2 * target_counts.get(1, 0)) if target_counts.get(1, 0) > 0 else 1.0
            }
            
            log.info(f"Using class weights: {self.class_weights}")
        else:
            self.class_weights = None
    
    def _setup_dataloaders(self):
        if self.config.use_train_validation_cols:
            training_rows = self.input_df.loc[self.input_df['training'] == 1]
            train_subjects = list(training_rows['anonymized_subject_id'])
            validation_rows = self.input_df.loc[self.input_df['validation'] == 1]
            val_subjects = list(validation_rows['anonymized_subject_id'])
        else:
            train_subjects, val_subjects = self.split_train_validation()

        self.train_dl = self.data_handler.init_dl(self.folder, train_subjects)
        self.val_dl = self.data_handler.init_dl(self.folder, val_subjects, is_val_set=True)

    
    def _setup_optimizer(self):
        if self.config.optimizer.lower() == 'adam':
            self.optimizer = Adam(
                self.model.parameters(), 
                lr=self.config.lr,
                weight_decay=self.config.weight_decay
            )
        else:
            self.optimizer = SGD(
                self.model.parameters(), 
                lr=self.config.lr, 
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
    
    def _setup_scheduler(self):
        if self.config.scheduler == 'step':
            self.scheduler = StepLR(self.optimizer, step_size=30, gamma=0.1)
        elif self.config.scheduler == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.epochs, eta_min=0)
        elif self.config.scheduler == 'onecycle':
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config.lr * 10,
                total_steps=len(self.train_dl) * self.config.epochs,
                pct_start=0.3
            )
        else:  # 'plateau'
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=0.1, 
                patience=10,
                verbose=True
            )
    
    def train(self):
        log.info("Starting training...")
        
        trainer = LogisticRegressionTrainer(
            self.model, 
            self.optimizer, 
            self.device, 
            self.class_weights,
            threshold=self.config.threshold
        )
        
        best_val_loss = float('inf')
        best_val_acc = 0.0
        best_epoch = 0
        
        for epoch in range(1, self.config.epochs + 1):
            # Training
            trn_metrics = trainer.train_epoch(epoch, self.train_dl)
            trn_loss = trn_metrics[METRICS_LOSS_NDX].mean().item()
            
            # Validation
            val_metrics = trainer.validate_epoch(epoch, self.val_dl)
            val_loss = val_metrics[METRICS_LOSS_NDX].mean().item()
            
            # Calculate validation accuracy
            val_acc = accuracy_score(
                val_metrics[METRICS_LABEL_NDX].numpy(), 
                val_metrics[METRICS_PRED_NDX].numpy()
            )
            
            # Log metrics
            self.tb_logger.log_metrics('trn', epoch, trn_metrics, trainer.total_samples)
            self.tb_logger.log_metrics('val', epoch, val_metrics, trainer.total_samples)
            
            # Step the scheduler based on validation loss
            if self.config.scheduler != 'onecycle':
                if self.config.scheduler == 'plateau':
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Track the best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_epoch = epoch
                
                # Save the best model
                if self.config.model_save_location:
                    log.info(f"New best validation accuracy: {best_val_acc:.4f}, saving model")
                    self.save_model(self.config.model_save_location)
            
            # Log epoch summary
            log.info(f"Epoch {epoch}/{self.config.epochs} - "
                    f"Train Loss: {trn_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val Accuracy: {val_acc:.4f}, "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        log.info(f"Training completed. Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
        
        # Save final model if needed
        if not self.config.model_save_location:
            final_model_path = f'./logistic_model_final-{self.time_str}.pt'
            self.save_model(final_model_path)
        
        # Generate predictions and plots
        if self.config.csv_output_file:
            self.generate_predictions()
        
        # Close tensorboard logger
        self.tb_logger.close()
    
    def save_model(self, path):
        """Save the trained model to disk"""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save(self.model.state_dict(), path)
        log.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load a trained model from disk"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        log.info(f"Model loaded from {path}")
    
    def generate_predictions(self):
        """Generate predictions for the entire dataset and save to CSV"""
        log.info("Generating predictions...")
        
        # Ensure the model is in evaluation mode
        self.model.eval()
        
        # Make a copy of the input dataframe
        output_df = self.df.copy()
        
        # Create dataset for the entire data
        full_dataset = LogisticRegressionDataset(
            self.df,
            self.config.features,
            self.config.target,
            normalize=self.config.normalize_features
        )
        
        # Create dataloader
        full_dl = DataLoader(
            full_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.use_cuda
        )
        
        # Generate predictions
        all_probs = []
        all_preds = []
        
        with torch.no_grad():
            for features, _ in full_dl:
                features = features.to(self.device)
                probs = self.model(features).squeeze(dim=-1).cpu().numpy()
                preds = (probs >= self.config.threshold).astype(int)
                
                all_probs.extend(probs)
                all_preds.extend(preds)
        
        # Add predictions to output dataframe
        output_df['probability'] = all_probs
        output_df['prediction'] = all_preds
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(self.config.csv_output_file) or '.', exist_ok=True)
        
        # Save to CSV
        output_df.to_csv(self.config.csv_output_file, index=False)
        log.info(f"Predictions saved to {self.config.csv_output_file}")
        
        # Generate evaluation metrics
        y_true = output_df[self.config.target].values
        y_pred = output_df['prediction'].values
        y_prob = output_df['probability'].values
        
        # Calculate and log metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        log.info(f"Final metrics on full dataset:")
        log.info(f"  Accuracy:  {accuracy:.4f}")
        log.info(f"  Precision: {precision:.4f}")
        log.info(f"  Recall:    {recall:.4f}")
        log.info(f"  F1 Score:  {f1:.4f}")
        
        # Only compute AUC if we have both classes
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_prob)
            log.info(f"  AUC:       {auc:.4f}")
        
        # Create confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
        log.info(f"Confusion Matrix:\n{conf_matrix}")
        
        # Create ROC curve plot if plot location is specified
        if self.config.plot_location:
            self.create_plots(y_true, y_pred, y_prob)
    
    def create_plots(self, y_true, y_pred, y_prob):
        """Create and save evaluation plots"""
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        im = ax1.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax1.set_title('Confusion Matrix')
        
        # Add labels to confusion matrix
        classes = ['Negative (0)', 'Positive (1)']
        tick_marks = np.arange(len(classes))
        ax1.set_xticks(tick_marks)
        ax1.set_yticks(tick_marks)
        ax1.set_xticklabels(classes)
        ax1.set_yticklabels(classes)
        
        # Add text annotations
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax1.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        # Add axis labels
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # Add colorbar
        plt.colorbar(im, ax=ax1)
        
        # Only create ROC curve if we have both classes
        if len(np.unique(y_true)) > 1:
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            
            ax2.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (area = {roc_auc:.2f})')
            ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax2.set_xlim([0.0, 1.0])
            ax2.set_ylim([0.0, 1.05])
            ax2.set_xlabel('False Positive Rate')
            ax2.set_ylabel('True Positive Rate')
            ax2.set_title('Receiver Operating Characteristic')
            ax2.legend(loc="lower right")
        else:
            ax2.text(0.5, 0.5, 'ROC curve cannot be created\n(only one class present)',
                    ha='center', va='center', fontsize=12)
            ax2.set_title('ROC Curve (Unavailable)')
        
        # Save the figure
        plt.tight_layout()
        os.makedirs(os.path.dirname(self.config.plot_location) or '.', exist_ok=True)
        plt.savefig(self.config.plot_location, dpi=300, bbox_inches='tight')
        log.info(f"Plots saved to {self.config.plot_location}")
        plt.close()
    

def main():
    log_reg_app = LogisticRegressionApp()
    log_reg_app.train()


if __name__ == "__main__":
    main()