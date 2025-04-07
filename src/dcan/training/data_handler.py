# Data Handler Class to manage dataset operations
import torch
from dcan.data_sets.dsets import LoesScoreDataset
from torch.utils.data import DataLoader


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
