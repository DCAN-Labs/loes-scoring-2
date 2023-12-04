# Author: Paul Reiners

import pandas as pd
import torch
import numpy as np

from dcan.training.training import get_subject_from_file_name, get_session_from_file_name, LoesScoringTrainingApp
from util.logconf import logging
from util.util import enumerateWithEstimate
from pathlib import Path

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)


class LeaveOneOutCrossValidation(LoesScoringTrainingApp):
    def run(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        df = pd.read_csv(self.cli_args.csv_data_file)
        df['subject'] = df.apply(lambda row: get_subject_from_file_name(row['file']), axis=1)
        df['session'] = df.apply(lambda row: get_session_from_file_name(row['file']), axis=1)
        df['prediction'] = np.nan

        output_df = df.copy()

        subjects = list(df.subject.unique())
        log.info(f'subject count: {len(subjects)}')
        n = 3
        test_subjects = subjects[:n]
        train_subjects = subjects[n:]
        log.info(f'train %: {int(100 * len(train_subjects) / len(subjects))}')
        self.leave_one_out(df, output_df, train_subjects, test_subjects)
        filepath = Path(self.cli_args.output_csv_file)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.output_df.sort_values(by=['subject', 'session'])
        self.output_df.to_csv(filepath)

    def leave_one_out(self, df, output_df, train_subjects, test_subjects):
        self.model = self.init_model()
        self.optimizer = self.init_optimizer()
        self.train_subjects = train_subjects.copy()
        self.val_subjects = test_subjects
        train_dl = self.init_train_dl(df, self.train_subjects, output_df)
        val_dl = self.init_val_dl(df, self.val_subjects, output_df)
        for epoch_ndx in range(1, self.cli_args.epochs + 1):
            self.train(epoch_ndx, train_dl, val_dl)
        if hasattr(self, 'trn_writer'):
            self.trn_writer.close()
            self.val_writer.close()
        if isinstance(self.model, torch.nn.DataParallel):
            self.model = self.model.module
        torch.save(self.model.state_dict(), self.cli_args.model_save_location)
        torch.cuda.empty_cache()
        self.model.eval()
        with torch.no_grad():
            batch_iter = enumerateWithEstimate(
                val_dl,
                "leave_out_one_cross_validation",
                start_ndx=val_dl.num_workers,
            )
            for batch_ndx, batch_tup in batch_iter:
                self.make_predictions(batch_tup)

    def make_predictions(self, batch_tup):
        input_t, label_t, subjects, sessions_str = batch_tup
        x = input_t.to(self.device, non_blocking=True)
        labels = label_t.to(self.device, non_blocking=True)
        outputs = self.model(x)
        predictions = self.get_actual(outputs).tolist()
        n = len(labels)
        for i in range(n):
            prediction = predictions[i]
            subject = subjects[i]
            session_str = sessions_str[i]
            index = \
                self.output_df[(self.output_df['subject'] == subject) &
                               (self.output_df['session'] == session_str)] \
                    .index
            self.output_df.loc[index, 'prediction'] = prediction
        return subjects

    def train(self, epoch_ndx, train_dl, val_dl):
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


if __name__ == '__main__':
    LeaveOneOutCrossValidation().run()
