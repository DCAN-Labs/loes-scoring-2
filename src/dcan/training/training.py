# Author: Paul Reiners
import argparse
import datetime
import os
from itertools import chain
from typing import List

import numpy as np
import pandas as pd
import scipy
import scipy.stats
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dcan.data_sets.dsets import LoesScoreDataset
from dcan.metrics import get_standardized_rmse
from dcan.plot.create_scatterplot import create_scatterplot
from dcan.inference.models import AlexNet3D
from util.logconf import logging
from util.util import enumerateWithEstimate

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)

# Used for computeBatchLoss and logMetrics to index into metrics_t/metrics_a
METRICS_LABEL_NDX = 0
METRICS_PRED_NDX = 1
METRICS_LOSS_NDX = 2
METRICS_SIZE = 3


def flatten_chain(matrix):
    return list(chain.from_iterable(matrix))


def get_subject_from_file_name(file_name):
    start_pos = file_name.find('sub')
    end_pos = file_name.find('ses', start_pos) - 1
    subject = file_name[start_pos:end_pos]

    return subject


def get_session_from_file_name(file_name):
    start_pos = file_name.find('ses')
    end_pos = file_name.find('_', start_pos)
    session = file_name[start_pos:end_pos]

    return session


class LoesScoringTrainingApp:
    def __init__(self, sys_argv=None):
        self.device = None
        self.cli_args = None
        self.use_cuda = False
        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0

        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--tb-prefix',
                                 default='loes_scoring',
                                 help="Data prefix to use for Tensorboard run. Defaults to loes_scoring.",
                                 )
        self.parser.add_argument('--csv-data-file',
                                 help="CSV data file.",
                                 )
        self.parser.add_argument('--output-csv-file',
                                 help="Output CSV data file.",
                                 )
        self.parser.add_argument('--num-workers',
                                 help='Number of worker processes for background data loading',
                                 default=8,
                                 type=int,
                                 )
        self.parser.add_argument('--batch-size',
                                 help='Batch size to use for training',
                                 default=32,
                                 type=int,
                                 )
        self.parser.add_argument('--epochs',
                                 help='Number of epochs to train for',
                                 default=1,
                                 type=int,
                                 )
        self.parser.add_argument('--file-path-column-index',
                                 help='The index of the file path in the CSV file',
                                 type=int,
                                 )
        self.parser.add_argument('--loes-score-column-index',
                                 help='The index of the Loes score in the CSV file',
                                 type=int,
                                 )
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.parser.add_argument('--model-save-location',
                                 help='Location to save models',
                                 default=f'./model-{self.time_str}.pt',
                                 )
        self.parser.add_argument('--plot-location',
                                 help='Location to save plot',
                                 )
        self.parser.add_argument('--optimizer',
                                 help="optimizer type.",
                                 default='Adam',
                                 )
        self.parser.add_argument('--model',
                                 help="Model type.",
                                 default='AlexNet',
                                 )
        self.parser.add_argument('comment',
                                 help="Comment suffix for Tensorboard run.",
                                 nargs='?',
                                 default='dcan',
                                 )
        self.parser.add_argument('--lr',
                                 help='Learning rate',
                                 default=0.001,
                                 type=float,
                                 )
        self.parser.add_argument('--gd',
                                 type=int,
                                 help="Use Gd-enhanced scans.")
        self.parser.add_argument('--use-train-validation-cols', action='store_true')
        self.parser.add_argument('-k', help='index for 5-fold validation', type=int, default=0)
        self.cli_args = self.parser.parse_args(sys_argv)
        self.df = pd.read_csv(self.cli_args.csv_data_file)
        self.output_df = None

        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.model = self.init_model()
        self.optimizer = self.init_optimizer()
        self.train_subjects = []
        self.val_subjects = []

    def init_model(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.cli_args.model == 'ResNet':
            log.info("Using ResNet3D")
            model = ResNet3D().to(device)
        else:
            model = AlexNet3D(4608)
        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)
        return model

    def init_optimizer(self):
        if self.cli_args.optimizer.lower() == 'adam':
            return Adam(self.model.parameters(), lr=self.cli_args.lr)
        else:
            return SGD(self.model.parameters(), lr=0.001, momentum=0.99)

    def init_train_dl(self, df, train_subjects, output_df):
        train_ds = LoesScoreDataset(train_subjects,
                                    df, output_df,
                                    is_val_set_bool=False,
                                    )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return train_dl

    def init_val_dl(self, df, val_subjects: List[str], output_df):
        val_ds = LoesScoreDataset(val_subjects, df,
                                  output_df,
                                  is_val_set_bool=True,
                                  )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return val_dl

    def init_tensorboard_writers(self):
        self.create_summary_writers()

    def create_summary_writers(self):
        if self.trn_writer is None:
            log_dir = os.path.join('runs', self.cli_args.tb_prefix, self.time_str)

            self.trn_writer = SummaryWriter(
                log_dir=log_dir + '-trn_cls-' + self.cli_args.comment)
            self.val_writer = SummaryWriter(
                log_dir=log_dir + '-val_cls-' + self.cli_args.comment)

    @staticmethod
    def get_actual(outputs):
        actual = outputs[0].squeeze(1)

        return actual

    def get_output_distributions(self):
        with torch.no_grad():
            val_dl = self.init_val_dl(self.df, self.val_subjects, None)
            self.model.eval()
            batch_iter = enumerateWithEstimate(
                val_dl,
                "get_output_distributions",
                start_ndx=val_dl.num_workers,
            )
            distributions = dict()
            for batch_ndx, batch_tup in batch_iter:
                input_t, label_t, subjects, sessions_str = batch_tup
                x = input_t.to(self.device, non_blocking=True)
                labels = label_t.to(self.device, non_blocking=True)
                outputs = self.model(x)
                predictions = self.get_actual(outputs).tolist()
                n = len(labels)
                for i in range(n):
                    label_int = int(labels[i].item())
                    if label_int not in distributions:
                        distributions[label_int] = []
                    prediction = predictions[i]
                    distributions[label_int].append(prediction)
                    subject = subjects[i]
                    session_str = sessions_str[i]
                    index = \
                        self.output_df[(self.output_df['subject'] == subject) &
                                       (self.output_df['session'] == session_str)]\
                            .index
                    self.output_df.loc[index, 'prediction'] = prediction

            return distributions

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        gd = self.cli_args.gd
        if gd in [0, 1]:
            self.df = self.df[self.df['Gd'] == gd]

        self.df['subject'] = self.df.apply(lambda row: get_subject_from_file_name(row['file']), axis=1)
        self.df['session'] = self.df.apply(lambda row: get_session_from_file_name(row['file']), axis=1)

        self.output_df = self.df.copy()
        self.output_df["training"] = np.nan
        self.output_df["validation"] = np.nan
        # TODO Add test column

        subjects = list(self.df.subject.unique())
        grouped = self.df.groupby('subject')['loes-score'].max()
        sorted_groups = grouped.sort_values(ascending=True)
        if not self.cli_args.use_train_validation_cols:
            i = 0
            train_subjects = []
            for item in sorted_groups.items():
                if i % 5 != self.cli_args.k:
                    train_subjects.append(item[0])
                i += 1
        else:
            train_df = self.df.loc[self.df['training'] == 1]
            train_subjects = train_df.subject.unique()

        val_subjects = [subject for subject in subjects if subject not in train_subjects]
        self.train_subjects.extend(train_subjects)
        self.val_subjects.extend(val_subjects)

        train_dl = self.init_train_dl(self.df, self.train_subjects, self.output_df)
        val_dl = self.init_val_dl(self.df, self.val_subjects, self.output_df)

        from pathlib import Path
        filepath = Path(self.cli_args.output_csv_file)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.output_df = self.output_df.astype({"training": int, "validation": int})
        self.output_df.sort_values(by=['subject', 'session'])
        self.output_df.to_csv(filepath)

        for epoch_ndx in range(1, self.cli_args.epochs + 1):
            log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                epoch_ndx,
                self.cli_args.epochs,
                len(train_dl),
                len(val_dl),
                self.cli_args.batch_size,
                (torch.cuda.device_count() if self.use_cuda else 1),
            ))

            trn_metrics_t = self.do_training(epoch_ndx, train_dl)
            log.debug(f'trn_metrics_t: {trn_metrics_t}')
            self.log_metrics(epoch_ndx, 'trn', trn_metrics_t)

            val_metrics_t = self.do_validation(epoch_ndx, val_dl)
            self.log_metrics(epoch_ndx, 'val', val_metrics_t)

        if hasattr(self, 'trn_writer'):
            self.trn_writer.close()
            self.val_writer.close()

        # save state dict of DataParallel object
        if isinstance(self.model, torch.nn.DataParallel):
            self.model = self.model.module
        torch.save(self.model.state_dict(), self.cli_args.model_save_location)

        output_distributions = self.get_output_distributions()
        actuals = []
        predictions = []
        for key, val in output_distributions.items():
            for v in val:
                actuals.append(key)
                predictions.append(v)

        try:
            standardized_rmse = get_standardized_rmse(actuals, predictions)
            log.info(f'standardized_rmse: {standardized_rmse}')
        except ZeroDivisionError as err:
            log.error(f'Could not compute standardized RMSE because sigma is 0: {err}')

        result = scipy.stats.linregress(actuals, predictions)

        self.output_df.sort_values(by=['subject', 'session'])
        self.output_df.to_csv(filepath, index=False)

        if self.cli_args.plot_location:
            validation_df = self.output_df[self.output_df['validation'] == 1]
            create_scatterplot(
                validation_df[['loes-score', 'prediction', 'subject', 'session']], self.cli_args.plot_location)

        # noinspection PyUnresolvedReferences
        log.info(f"correlation:    {result.rvalue}")
        # noinspection PyUnresolvedReferences
        log.info(f"p-value:        {result.pvalue}")
        # noinspection PyUnresolvedReferences
        log.info(f"standard error: {result.stderr}")

    def do_training(self, epoch_ndx, train_dl):
        self.model.train()
        trn_metrics_g = torch.zeros(
            METRICS_SIZE,
            len(train_dl.dataset),
            device=self.device,
        )

        batch_iter = enumerateWithEstimate(
            train_dl,
            "E{} Training".format(epoch_ndx),
            start_ndx=train_dl.num_workers,
        )
        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()

            loss_var = self.compute_batch_loss(
                batch_ndx,
                batch_tup,
                train_dl.batch_size,
                trn_metrics_g
            )
            log.debug(f'trn_metrics_g: {trn_metrics_g}')

            loss_var.backward()
            self.optimizer.step()

        self.totalTrainingSamples_count += len(train_dl.dataset)

        return trn_metrics_g.to('cpu')

    def do_validation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            self.model.eval()
            val_metrics_g = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device,
            )

            batch_iter = enumerateWithEstimate(
                val_dl,
                "E{} Validation ".format(epoch_ndx),
                start_ndx=val_dl.num_workers,
            )
            for batch_ndx, batch_tup in batch_iter:
                self.compute_batch_loss(
                    batch_ndx, batch_tup, val_dl.batch_size, val_metrics_g)

        return val_metrics_g.to('cpu')

    def compute_batch_loss(self, batch_ndx, batch_tup, batch_size, metrics_g):
        input_t, label_t, _, _ = batch_tup

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        outputs_g = self.model(input_g)

        loss_func = nn.MSELoss(reduction='none')
        loss_g = loss_func(
            outputs_g[0].squeeze(1),
            label_g,
        )
        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)

        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = \
            label_g.detach()
        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = \
            loss_g.detach()

        return loss_g.mean()

    def log_metrics(
            self,
            epoch_ndx,
            mode_str,
            metrics_t,
    ):
        self.init_tensorboard_writers()
        log.info("E{} {}".format(
            epoch_ndx,
            type(self).__name__,
        ))

        metrics_dict = {'loss/all': metrics_t[METRICS_LOSS_NDX].mean()}

        log.info(
            ("E{} {:8} {loss/all:.4f} loss, "
             ).format(
                epoch_ndx,
                mode_str,
                **metrics_dict,
            )
        )

        writer = getattr(self, mode_str + '_writer')

        for key, value in metrics_dict.items():
            writer.add_scalar(key, value, self.totalTrainingSamples_count)


if __name__ == '__main__':
    LoesScoringTrainingApp().main()
