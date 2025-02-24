import logging

import pandas as pd
import statistics
import torch
import torchio as tio
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from dcan.inference.models import AlexNet3D
from monai.networks.nets import Regressor

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)


def load_model(model_name, model_save_location, device='cpu'):
    if model_name == 'ResNet':
        model = Regressor(in_shape=[1, 197, 233, 189], out_shape=1, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2))
        log.info("Using ResNet")
    else:
        model = AlexNet3D(4608)
        log.info("Using AlexNet3D")
    model.to(device)

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


def get_validation_info(model_type, model_save_location, input_csv_location):
    model = load_model(model_type, model_save_location, device='cpu')

    df = pd.read_csv(input_csv_location)
    validation_rows = df.loc[df['validation'] == 1]
    output_df = validation_rows.copy()
    subjects = list(output_df['anonymized_subject_id'])
    sessions = list(output_df['anonymized_session_id'])
    actual_scores = list(output_df['loes-score'])
    with torch.no_grad():
        inputs = list(output_df.apply(predict, axis=1))

        predictions = [model(input) for input in inputs]
        predict_vals = [p[0].item() for p in predictions]

        return subjects, sessions, actual_scores, predict_vals



def compute_standardized_rmse(actual_scores, predict_vals):
    rmse = compute_rmse(predict_vals, actual_scores)
    sigma = statistics.stdev(actual_scores)
    standardized_rmse = rmse / sigma
    
    return standardized_rmse


def create_correlation_coefficient(actual_vals, predicted_vals):
    x = np.array(actual_vals)
    y = np.array(predicted_vals)

    correlation_matrix = np.corrcoef(x, y)
    correlation_coefficient = correlation_matrix[0, 1]

    return correlation_coefficient


def create_scatter_plot(actual_vals, predicted_vals, output_file):
    fig, ax = plt.subplots()
    ax.scatter(actual_vals, predicted_vals, s=25, c='blue', cmap=plt.cm.coolwarm, zorder=10)

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Add labels and title
    plt.xlabel("Actual Loes score")
    plt.ylabel("Predicted Loes score")
    plt.title("Loes score prediction")

    # Save the plot to a file
    plt.savefig(output_file, dpi=300) 

def get_predicted_value(row, subjects, sessions, predict_vals):
    zipped_data = zip(subjects, sessions, predict_vals)

    for subject, session, predict_val in zipped_data:
        if row['anonymized_subject_id'] == subject and row['anonymized_session_id'] == session:
            return predict_val
    return np.nan

def add_predicted_values(subjects, sessions, predict_vals, input_csv_location):
    input_df = pd.read_csv(input_csv_location) 
    output_df = input_df.copy() 
    output_df['predicted_loes_score'] = output_df.apply(get_predicted_value, axis=1, args=(subjects, sessions, predict_vals))

    return output_df
