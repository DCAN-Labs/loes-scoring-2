import argparse
import datetime
import os
import statistics
from math import sqrt

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from reprex.models import AlexNet3D_Dropout_Regression
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


class TrainingApp:
    def __init__(self):
        self.device = None
        self.cli_args = None
        self.use_cuda = False
        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.model = self.init_model()
        self.optimizer = self.init_optimizer()

    def init_model(self):
        model = AlexNet3D_Dropout_Regression(3456)
        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)

        return model

    def init_optimizer(self):
        # return SGD(self.model.parameters(), lr=0.001, momentum=0.99)
        return Adam(self.model.parameters())

    def get_args(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--tb-prefix',
                            default='dcan',
                            help="Data prefix to use for Tensorboard run. Defaults to dcan.",
                            )
        parser.add_argument('--csv-data-file',
                            help="CSV data file.",
                            )
        parser.add_argument('comment',
                            help="Comment suffix for Tensorboard run.",
                            nargs='?',
                            default='loes_scoring',
                            )
        parser.add_argument('--model',
                            help="Model type.",
                            default='AlexNet',
                            )
        # See https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference
        parser.add_argument('--model-save-location',
                            help="Where to save the trained model.",
                            default=f'./model-{self.time_str}.pt',
                            )
        parser.add_argument('--num-workers',
                            help='Number of worker processes for background data loading',
                            default=8,
                            type=int,
                            )
        parser.add_argument('--batch-size',
                            help='Batch size to use for training',
                            default=32,
                            type=int,
                            )
        parser.add_argument('--epochs',
                            help='Number of epochs to train for',
                            default=1,
                            type=int,
                            )
        parser.add_argument('--optimizer',
                            help="optimizer type.",
                            default='Adam',
                            )

        return parser.parse_args()

    @staticmethod
    def get_actual(outputs):
        actual = outputs[0].squeeze(1)

        return actual

    def init_val_dl(self, csv_data_file=None):
        pass

    def get_labels_and_predictions(self, batch_tup):
        input_t, label_t = batch_tup
        x = input_t.to(self.device, non_blocking=True)
        labels = label_t.to(self.device, non_blocking=True)
        outputs = self.model(x)
        predictions = self.get_actual(outputs).tolist()
        n = len(labels)
        return labels, n, predictions

    def get_mean_and_sigma(self, val_dl):
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
        return rmse, sigma

    def init_tensorboard_writers(self):
        if self.trn_writer is None:
            log_dir = os.path.join('runs', self.cli_args.tb_prefix, self.time_str)

            self.trn_writer = SummaryWriter(
                log_dir=log_dir + '-trn_reg-' + self.cli_args.comment)
            self.val_writer = SummaryWriter(
                log_dir=log_dir + '-val_reg-' + self.cli_args.comment)

    def log_metrics(
            self,
            epoch_ndx,
            mode_str,
            metrics_t,
    ):
        self.init_tensorboard_writers()
        self.log_epoch_metrics(epoch_ndx, metrics_t, mode_str)

    def compute_batch_loss(self, batch_ndx, batch_tup, batch_size, metrics_g):
        input_t, label_t = batch_tup

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        self.model = self.model.to(self.device, non_blocking=True)
        outputs_g = self.model(input_g)

        def motion_qc_loss_with_weight(weight):
            def motion_qc_loss(y_true, y_pred):
                loss_func = nn.MSELoss(reduction='none')
                loss = loss_func(y_true, y_pred)
                for i in range(len(loss)):
                    if y_true[i] <= 1.1 or y_true[i] >= 3.9:
                        loss[i] = loss[i] * weight

                return loss
            return motion_qc_loss

        loss_g = motion_qc_loss_with_weight(4.0)(
            label_g,
            outputs_g[0].squeeze(1),
        )
        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)

        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = \
            label_g.detach()
        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = \
            loss_g.detach()

        mean_loss = loss_g.mean()

        return mean_loss

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

            loss = self.compute_batch_loss(
                batch_ndx,
                batch_tup,
                train_dl.batch_size,
                trn_metrics_g
            )

            loss.backward()
            self.optimizer.step()

            # # This is for adding the model graph to TensorBoard.
            # if epoch_ndx == 1 and batch_ndx == 0:
            #     with torch.no_grad():
            #         model = LunaModel()
            #         self.trn_writer.add_graph(model, batch_tup[0], verbose=True)
            #         self.trn_writer.close()

        self.totalTrainingSamples_count += len(train_dl.dataset)

        return trn_metrics_g.to('cpu')

    def run_epochs(self, train_dl, val_dl):
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

    def log_epoch_metrics(self, epoch_ndx, metrics_t, mode_str):
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
