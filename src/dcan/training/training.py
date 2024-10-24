import argparse
import datetime
import pandas as pd
import torch
import logging

from dcan.data_sets.dsets import LoesScoreDataset
from dcan.inference.models import AlexNet3D
from faimed3d.models.resnet import ResNet3D
from torch.optim import Adam
from torch.utils.data import DataLoader
import os
from torch.optim.sgd import SGD
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.nn as nn


from util.util import enumerateWithEstimate




log = logging.getLogger(__name__)

# Used for computeBatchLoss and logMetrics to index into metrics_t/metrics_a
METRICS_LABEL_NDX = 0
METRICS_PRED_NDX = 1
METRICS_LOSS_NDX = 2
METRICS_SIZE = 3




class LoesScoringTrainingApp:
    def __init__(self, sys_argv=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cli_args = self.parse_cli_args(sys_argv)
        self.df = pd.read_csv(self.cli_args.csv_data_file)
        self.output_df = self.df.copy()
        self.trn_writer = None
        self.val_writer = None
        self.train_subjects = []
        self.val_subjects = []
        self.model = self.init_model()
        self.optimizer = self.init_optimizer()
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.total_training_samples = 0

    def parse_cli_args(self, sys_argv):
        parser = argparse.ArgumentParser()
        parser.add_argument('--tb-prefix', default='loes_scoring', help="Tensorboard run prefix.")
        parser.add_argument('--csv-data-file', help="CSV data file.")
        parser.add_argument('--output-csv-file', help="Output CSV data file.")
        parser.add_argument('--num-workers', type=int, default=8, help='Worker processes for data loading.')
        parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training.')
        parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train.')
        parser.add_argument('--file-path-column-index', type=int)
        parser.add_argument('--loes-score-column-index', type=int)
        parser.add_argument('--model-save-location', default=f'./model-{self.time_str}.pt', help='Model save path.')
        parser.add_argument('--plot-location', help='Location to save plot.')
        parser.add_argument('--optimizer', default='Adam', help="Optimizer type (Adam or SGD).")
        parser.add_argument('--model', default='AlexNet', help="Model type (AlexNet or ResNet).")
        parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
        parser.add_argument('--use-train-validation-cols', action='store_true')
        parser.add_argument('-k', type=int, default=0, help='Index for 5-fold validation.')
        return parser.parse_args(sys_argv)

    def init_model(self):
        log.info(f"Initializing model: {self.cli_args.model}")
        model = AlexNet3D(4608) if self.cli_args.model == 'AlexNet' else ResNet3D()
        return model.to(self.device)

    def init_optimizer(self):
        optimizer_type = self.cli_args.optimizer.lower()
        optimizer_cls = Adam if optimizer_type == 'adam' else SGD
        return optimizer_cls(self.model.parameters(), lr=self.cli_args.lr)

    def create_data_loader(self, subjects, is_val_set=False):
        dataset = LoesScoreDataset(subjects, self.df, self.output_df, is_val_set_bool=is_val_set)
        batch_size = self.cli_args.batch_size * (torch.cuda.device_count() if self.use_cuda else 1)
        return DataLoader(dataset, batch_size=batch_size, num_workers=self.cli_args.num_workers, pin_memory=self.use_cuda)

    def init_tensorboard_writers(self):
        if not self.trn_writer or not self.val_writer:
            log_dir = os.path.join('runs', self.cli_args.tb_prefix, self.time_str)
            self.trn_writer = SummaryWriter(log_dir=f"{log_dir}-train")
            self.val_writer = SummaryWriter(log_dir=f"{log_dir}-val")

    def main(self):
        log.info(f"Starting training for {self.cli_args.epochs} epochs")
        self.output_df["training"] = np.nan
        self.output_df["validation"] = np.nan

        # Split train/validation
        self.split_train_validation()

        # Initialize data loaders
        train_dl = self.create_data_loader(self.train_subjects)
        val_dl = self.create_data_loader(self.val_subjects, is_val_set=True)

        # Train and validate across epochs
        for epoch in range(1, self.cli_args.epochs + 1):
            log.info(f"Epoch {epoch}/{self.cli_args.epochs}")
            train_metrics = self.run_epoch(epoch, train_dl, is_train=True)
            val_metrics = self.run_epoch(epoch, val_dl, is_train=False)
            self.log_metrics(epoch, 'train', train_metrics)
            self.log_metrics(epoch, 'val', val_metrics)

        # Save model and generate outputs
        torch.save(self.model.state_dict(), self.cli_args.model_save_location)
        self.generate_output()

    def run_epoch(self, epoch, data_loader, is_train):
        mode = 'train' if is_train else 'validation'
        self.model.train() if is_train else self.model.eval()
        metrics = torch.zeros(METRICS_SIZE, len(data_loader.dataset), device=self.device)
        for batch_idx, batch in enumerateWithEstimate(data_loader, f"E{epoch} {mode.capitalize()}"):
            loss = self.compute_batch_loss(batch_idx, batch, data_loader.batch_size, metrics)
            if is_train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return metrics.to('cpu')

    def compute_batch_loss(self, batch_idx, batch, batch_size, metrics):
        inputs, labels, *_ = batch
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        outputs = self.model(inputs)
        loss = nn.MSELoss(reduction='none')(outputs.squeeze(1), labels)

        start, end = batch_idx * batch_size, batch_idx * batch_size + labels.size(0)
        metrics[METRICS_LABEL_NDX, start:end] = labels.detach()
        metrics[METRICS_LOSS_NDX, start:end] = loss.detach()
        return loss.mean()
