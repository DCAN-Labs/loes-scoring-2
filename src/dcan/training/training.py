# Author: Paul Reiners
import argparse
import datetime
import os
import statistics
from math import sqrt

import scipy
import scipy.stats
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dcan.data.partial_loes_scores import get_partial_loes_scores
from dcan.data_sets.dsets import LoesScoreDataset
from dcan.plot.create_scatterplot import create_scatterplot
from reprex.models import AlexNet3D
from util.logconf import logging
from util.util import enumerateWithEstimate

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

# Used for computeBatchLoss and logMetrics to index into metrics_t/metrics_a
METRICS_LABEL_NDX = 0
METRICS_PRED_NDX = 1
METRICS_LOSS_NDX = 2
METRICS_SIZE = 3


def compute_pearson_correlation_coefficient(d):
    xs = []
    ys = []
    for x in d:
        vals = d[x]
        for y in vals:
            xs.append(x)
            ys.append(y)
    result = scipy.stats.linregress(xs, ys)

    return result


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
        self.parser.add_argument('--anatomical-region',
                                 help="Anatomical region to train on.",
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
        self.parser.add_argument('--use-gd-only',
                                 help='Use Gadolinium-enhanced scans only',
                                 default=0,
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
        self.cli_args = self.parser.parse_args(sys_argv)
        partial_scores_file_path = \
            '/home/miran045/reine097/projects/loes-scoring-2/data/9_7 MRI sessions Igor Loes score updated.csv'
        self.partial_loes_scores = get_partial_loes_scores(partial_scores_file_path)

        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.model = self.init_model()
        self.optimizer = self.init_optimizer()

    def init_model(self):
        model = AlexNet3D(4608)
        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)
        return model

    def init_optimizer(self):
        # return SGD(self.model.parameters(), lr=0.001, momentum=0.99)
        return Adam(self.model.parameters())

    def init_train_dl(self, csv_data_file):
        train_ds = LoesScoreDataset(
            csv_data_file,
            use_gd_only=self.cli_args.use_gd_only,
            val_stride=10,
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

    def init_val_dl(self, csv_data_file):
        val_ds = LoesScoreDataset(
            csv_data_file,
            use_gd_only=self.cli_args.use_gd_only,
            val_stride=10,
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

    def get_standardized_rmse(self):
        with torch.no_grad():
            val_dl = self.init_val_dl(self.cli_args.csv_data_file)
            self.model.eval()
            batch_iter = enumerateWithEstimate(
                val_dl,
                "get_standardized_rmse",
                start_ndx=val_dl.num_workers,
            )
            squares_list = []
            prediction_list = []
            for batch_ndx, batch_tup in batch_iter:
                input_t, label_t = batch_tup
                x = input_t.to(self.device, non_blocking=True)
                labels = label_t.to(self.device, non_blocking=True)
                outputs = self.model(x)
                actual = self.get_actual(outputs)
                prediction_list.extend(actual.tolist())
                difference = torch.subtract(labels, actual)
                squares = torch.square(difference)
                squares_list.extend(squares.tolist())
            rmse = sqrt(sum(squares_list) / len(squares_list))
            sigma = statistics.stdev(prediction_list)

            return rmse / sigma

    def get_output_distributions(self):
        with torch.no_grad():
            val_dl = self.init_val_dl(self.cli_args.csv_data_file)
            self.model.eval()
            batch_iter = enumerateWithEstimate(
                val_dl,
                "get_output_distributions",
                start_ndx=val_dl.num_workers,
            )
            distributions = dict()
            for batch_ndx, batch_tup in batch_iter:
                input_t, label_t = batch_tup
                x = input_t.to(self.device, non_blocking=True)
                labels = label_t.to(self.device, non_blocking=True)
                outputs = self.model(x)
                predictions = self.get_actual(outputs).tolist()
                n = len(labels)
                for i in range(n):
                    label_int = int(labels[i].item())
                    if label_int not in distributions:
                        distributions[label_int] = []
                    distributions[label_int].append(predictions[i])

            return distributions

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        train_dl = self.init_train_dl(self.cli_args.csv_data_file)
        val_dl = self.init_val_dl(self.cli_args.csv_data_file)

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

        try:
            standardized_rmse = self.get_standardized_rmse()
            log.info(f'standardized_rmse: {standardized_rmse}')
        except ZeroDivisionError as err:
            log.error(f'Could not compute stanardized RMSE because sigma is 0: {err}')

        output_distributions = self.get_output_distributions()
        if self.cli_args.plot_location:
            create_scatterplot(output_distributions, self.cli_args.plot_location)

        result = compute_pearson_correlation_coefficient(output_distributions)

        log.info(f"correlation:    {result.rvalue}")
        log.info(f"p-value:        {result.pvalue}")
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
        input_t, label_t = batch_tup

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
