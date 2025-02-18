import logging
import math
import os.path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import pandas as pd
import statistics
import sys
import torch
import torchio as tio
from math import sqrt

from dcan.data_sets.dsets import LoesScoreDataset
from dcan.inference.models import AlexNet3D

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)


def load_model(model_save_location, device='cpu'):
    model = AlexNet3D(4608).to(device)
    model.load_state_dict(torch.load(model_save_location, map_location=device))
    model.eval()
    return model
    

def predict(row):
    subject = row['anonymized_subject_id']
    session = row['anonymized_session_id']
    mprage_path = f'/home/feczk001/shared/projects/S1067_Loes/data/Fairview-ag/05-training_ready/{subject}_{session}_space-MNI_brain_mprage_RAVEL.nii.gz'
    mprage_image = tio.ScalarImage(mprage_path)
    transform = tio.Compose([
        tio.ToCanonical(),
        tio.ZNormalization(masking_method=tio.ZNormalization.mean),
    ])
    transformed_mprage_image = transform(mprage_image)
    mprage_image_tensor = transformed_mprage_image.data
    value = mprage_image_tensor.unsqueeze(0)

    return value


import torch.nn.functional as F

def compute_rmse(predictions, actuals):
    predictions_tensor = torch.tensor(predictions, dtype=torch.float32)
    actuals_tensor = torch.tensor(actuals, dtype=torch.float32)
    mse = F.mse_loss(predictions_tensor, actuals_tensor)
    return torch.sqrt(mse).item()


def compute_standardized_rmse(model_save_location, input_csv_location):
    model = load_model(model_save_location, device='cpu')

    df = pd.read_csv(input_csv_location)
    validation_rows = df.loc[df['validation'] == 1]
    output_df = validation_rows.copy()
    predictions = []
    actual_scores = list(output_df['loes-score'])
    with torch.no_grad():
        inputs = list(output_df.apply(predict, axis=1))

        predictions = [model(input) for input in inputs]
        predict_vals = [p[0].item() for p in predictions]
        rmse = compute_rmse(predict_vals, actual_scores)
        sigma = statistics.stdev(actual_scores)
        standardized_rmse = rmse / sigma
        
        return standardized_rmse


if __name__ == "__main__":
    standardized_rmse = compute_standardized_rmse("/home/feczk001/shared/data/AlexNet/LoesScoring/loes_scoring_12.pt", 
                 "/users/9/reine097/projects/loes-scoring-2/data/anon_train_scans_and_loes_training_test_non_gd.csv")
    log.info(f'standardized_rmse: {standardized_rmse}')
