import logging

import pandas as pd
import statistics
import torch
import torchio as tio
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

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


def get_validation_info(model_save_location, input_csv_location):
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

        return actual_scores, predict_vals



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


def create_scatter_plot(actual_vals, predicted_vals):
    fig, ax = plt.subplots()
    ax.scatter(actual_vals, predicted_vals, s=25, cmap=plt.cm.coolwarm, zorder=10)

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
    plt.savefig("loes_score.png", dpi=300)  


if __name__ == "__main__":
    model_save_location = "/home/feczk001/shared/data/AlexNet/LoesScoring/loes_scoring_12.pt"
    input_csv_location = "/users/9/reine097/projects/loes-scoring-2/data/anon_train_scans_and_loes_training_test_non_gd.csv"
    actual_scores, predict_vals = get_validation_info(model_save_location, input_csv_location)
    standardized_rmse = \
        compute_standardized_rmse(actual_scores, predict_vals)
    print(f'standardized_rmse: {standardized_rmse}')
    create_scatter_plot(actual_scores, predict_vals)
    correlation_coefficient = create_correlation_coefficient(actual_scores, predict_vals)
    print(f'correlation_coefficient: {correlation_coefficient}')

    correlation, p_value = stats.pearsonr(actual_scores, predict_vals)
    print("Pearson correlation p-value:", p_value)

    correlation, p_value = stats.spearmanr(actual_scores, predict_vals)
    print("Spearman correlation p-value:", p_value)
