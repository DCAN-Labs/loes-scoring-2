import argparse
import datetime
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dcan.data_sets.dsets import LoesScoreDataset
from dcan.inference.models import AlexNet3D
from faimed3d.models.resnet import ResNet3D
from util.logconf import logging
from util.util import enumerateWithEstimate

# This script is a comprehensive deep learning pipeline for training a model to 
# predict Loes scores from MRI scans. It includes data loading, model selection,
# raining/validation loops, logging, and model saving.


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
        self.parser.add_argument('--csv-data-file', help="CSV data file.")
        self.parser.add_argument('--output-csv-file', help="Output CSV data file.")
        self.parser.add_argument('--num-workers', default=8, type=int, help='Number of worker processes')
        self.parser.add_argument('--batch-size', default=32, type=int, help='Batch size for training')
        self.parser.add_argument('--epochs', default=1, type=int, help='Number of epochs to train')
        self.parser.add_argument('--file-path-column-index', type=int, help='Index of the file path in CSV file')
        self.parser.add_argument('--loes-score-column-index', type=int, help='Index of the Loes score in CSV file')
        self.parser.add_argument('--model-save-location', default=f'./model-{datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")}.pt')
        self.parser.add_argument('--plot-location', help='Location to save plot')
        self.parser.add_argument('--optimizer', default='Adam', help="Optimizer type.")
        self.parser.add_argument('--model', default='AlexNet', help="Model type.")
        self.parser.add_argument('comment', nargs='?', default='dcan', help="Comment for Tensorboard run")
        self.parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
        self.parser.add_argument('--gd', type=int, help="Use Gd-enhanced scans.")
        self.parser.add_argument('--use-train-validation-cols', action='store_true')
        self.parser.add_argument('-k', type=int, default=0, help='Index for 5-fold validation')
        self.parser.add_argument('--folder', help='Folder where MRIs are stored')

    def parse_args(self, sys_argv):
        return self.parser.parse_args(sys_argv)


# Data Handler Class to manage dataset operations
class DataHandler:
    def __init__(self, df, output_df, use_cuda, batch_size, num_workers):
        self.df = df
        self.output_df = output_df
        self.use_cuda = use_cuda
        self.batch_size = batch_size
        self.num_workers = num_workers

    def init_dl(self, folder, subjects, is_val_set: bool = False):
        dataset = LoesScoreDataset(folder, subjects, self.df, self.output_df, is_val_set_bool=is_val_set)
        batch_size = self.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        return DataLoader(dataset, batch_size=batch_size, num_workers=self.num_workers, pin_memory=self.use_cuda)


# Model Handler Class to manage model operations
class ModelHandler:
    def __init__(self, model_name, use_cuda, device):
        self.model_name = model_name
        self.use_cuda = use_cuda
        self.device = device
        self.model = self._init_model()

    def _init_model(self):
        if self.model_name == 'ResNet':
            model = ResNet3D().to(self.device)
            log.info("Using ResNet3D")
        else:
            model = AlexNet3D(4608).to(self.device)
            log.info("Using AlexNet3D")

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


# Training/Validation Loop Handler
class TrainingLoop:
    def __init__(self, model_handler, optimizer, device):
        self.model_handler = model_handler
        self.optimizer = optimizer
        self.device = device
        self.total_samples = 0

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

    def _compute_batch_loss(self, batch_ndx, batch_tup, batch_size, metrics_g):
        input_t, label_t, _, _ = batch_tup
        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        outputs_g = self.model_handler.model(input_g)
        loss_func = nn.MSELoss(reduction='none')
        loss_g = loss_func(outputs_g[0].squeeze(1), label_g)

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)
        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = label_g.detach()
        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss_g.detach()

        return loss_g.mean()


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


# Main Application Class
class LoesScoringTrainingApp:
    def __init__(self, sys_argv=None):
        self.config = Config().parse_args(sys_argv)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_cuda = torch.cuda.is_available()

        self.df = pd.read_csv(self.config.csv_data_file)
        self.output_df = self.df.copy()

        self.model_handler = ModelHandler(self.config.model, self.use_cuda, self.device)
        self.optimizer = self._init_optimizer()

        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.tb_logger = TensorBoardLogger(self.config.tb_prefix, self.time_str, self.config.comment)

        self.data_handler = DataHandler(self.df, self.output_df, self.use_cuda, self.config.batch_size, self.config.num_workers)
        self.folder = self.config.folder

    def _init_optimizer(self):
        optimizer_type = self.config.optimizer.lower()
        optimizer_cls = Adam if optimizer_type == 'adam' else SGD
        return optimizer_cls(self.model_handler.model.parameters(), lr=self.config.lr)

    def main(self):
        log.info("Starting training...")
        self.output_df["training"] = np.nan
        self.output_df["validation"] = np.nan

        if self.config.gd == 0:
            self.df = self.df[~self.df['scan'].str.contains('Gd')]

        train_subjects, val_subjects = self.split_train_validation()
        train_dl = self.data_handler.init_dl(self.folder, train_subjects)
        val_dl = self.data_handler.init_dl(self.folder, val_subjects, is_val_set=True)

        loop_handler = TrainingLoop(self.model_handler, self.optimizer, self.device)
        
        for epoch in range(1, self.config.epochs + 1):
            log.info(f"Epoch {epoch}/{self.config.epochs}")

            trn_metrics = loop_handler.train_epoch(epoch, train_dl)
            val_metrics = loop_handler.validate_epoch(epoch, val_dl)

            self.tb_logger.log_metrics('trn', epoch, trn_metrics, loop_handler.total_samples)
            self.tb_logger.log_metrics('val', epoch, val_metrics, loop_handler.total_samples)

        self.model_handler.save_model(self.config.model_save_location)
        log.info(f"Model saved to {self.config.model_save_location}")
        self.tb_logger.close()

    def split_train_validation(self):
        all_users = self.df['anonymized_subject_id'].unique()
        self.df = self.df.sort_values('loes-score')
        validation_users = all_users[::5]
        training_users = [user for user in all_users if user not in validation_users]
        
        return training_users, validation_users

def main():
    loesScoringTrainingApp = LoesScoringTrainingApp(sys_argv=sys.argv)
    loesScoringTrainingApp.main()

if __name__ == "__main__":
    main()
