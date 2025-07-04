import logging
import sys

import pandas as pd
import statistics
import torch
import matplotlib.pyplot as plt
import numpy as np

import glob
import os
import torch.nn.functional as F
import nibabel as nib

from dcan.inference.models import AlexNet3D
from dcan.models.ResNet import get_resnet_model

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)


def load_model(model_name, model_save_location, device='cpu'):
    if model_name.lower() == 'resnet':
        model = get_resnet_model()
        log.info("Using ResNet")
    else:
        model = AlexNet3D(4608)
        log.info("Using AlexNet3D")
    model.to(device)

    model.load_state_dict(torch.load(model_save_location, map_location=device, weights_only=True))
    model.eval()

    return model
    

def predict(row, data_folder):
    subject = row['anonymized_subject_id']
    session = row['anonymized_session_id']
    mprage_path = f'{data_folder}/{subject}_{session}_space-MNI_brain_mprage_RAVEL.nii.gz'
    mprage_image_tensor = get_image_tensor(mprage_path)
    value = mprage_image_tensor.unsqueeze(0)

    return value

def z_normalize(image, mask=None):
    """Z-normalization (standardization) of image data"""
    if mask is not None:
        masked_data = image[mask > 0]
        mean = np.mean(masked_data)
        std = np.std(masked_data)
    else:
        mean = np.mean(image)
        std = np.std(image)
    
    if std == 0:
        return image - mean
    return (image - mean) / std


def get_image_tensor(mprage_path):
    # Load NIfTI file using nibabel instead of TorchIO
    nii_img = nib.load(mprage_path)
    image_data = nii_img.get_fdata()
    
    # Convert to float32 and ensure it's a numpy array
    image_data = np.array(image_data, dtype=np.float32)
    
    # Z-normalization (equivalent to tio.ZNormalization with mean masking)
    image_data = z_normalize(image_data)
    
    # Convert to torch tensor and add channel dimension
    mprage_image_tensor = torch.from_numpy(image_data).unsqueeze(0)  # Add channel dim
    
    # Move to CPU (though it's already on CPU)
    input_g = mprage_image_tensor.to('cpu', non_blocking=True)
    
    return input_g


def compute_rmse(predictions, actuals):
    predictions_tensor = torch.tensor(predictions, dtype=torch.float32)
    actuals_tensor = torch.tensor(actuals, dtype=torch.float32)
    mse = F.mse_loss(predictions_tensor, actuals_tensor)
    return torch.sqrt(mse).item()


def get_validation_info(model_type, model_save_location, input_csv_location, val_subjects, data_folder):
    model = load_model(model_type, model_save_location, device='cpu')

    df = pd.read_csv(input_csv_location)
    validation_rows = df[df['anonymized_subject_id'].isin(val_subjects)]
    output_df = validation_rows.copy()
    subjects = list(output_df['anonymized_subject_id'])
    sessions = list(output_df['anonymized_session_id'])
    actual_scores = list(output_df['loes-score'])
    with torch.no_grad():
        inputs = list(output_df.apply(predict, axis=1, args=(data_folder,)))

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
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Color by prediction error
    errors = np.abs(np.array(actual_vals) - np.array(predicted_vals))
    scatter = ax.scatter(actual_vals, predicted_vals, s=30, c=errors, 
                        cmap=plt.cm.Reds, alpha=0.7, zorder=10)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Prediction Error', fontsize=11)
    
    # Perfect prediction line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, 'k--', alpha=0.8, linewidth=2)
    
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    
    # Labels
    ax.set_xlabel("Actual Loes score")
    ax.set_ylabel("Predicted Loes score")
    ax.set_title("Loes score prediction")
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


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


def get_files_by_pattern(directory, pattern):
    """
    Retrieves all files in a directory matching a specified filename pattern.

    Args:
        directory (str): The path to the directory to search.
        pattern (str): The filename pattern to match (e.g., "*.txt", "image_*.png").

    Returns:
        list: A list of file paths that match the pattern.
    """
    search_pattern = os.path.join(directory, pattern)
    files = glob.glob(search_pattern)
    return files

def get_filename_from_path(file_path):
  """
  Extracts the filename from a given file path.

  Args:
    file_path: The path to the file.

  Returns:
    The filename, or None if the path is invalid.
  """
  return os.path.basename(file_path)


def make_predictions_on_folder(directory_path, file_pattern, model):
    matching_files = get_files_by_pattern(directory_path, file_pattern)

    df = pd.DataFrame({'anonymized_subject_id': [], 'anonymized_session_id': [], 'predicted_score': []})
    if matching_files:
        for file_path in matching_files:
            image_tensor = get_image_tensor(file_path)
            file_name = get_filename_from_path(file_path)
            parts = file_name.split('_')
            anonymized_subject_id = parts[0]
            anonymized_session_id = parts[1]
            with torch.no_grad():
                unsqueezed_image_tensor = image_tensor.unsqueeze(0)
                prediction = model(unsqueezed_image_tensor)
                prediction_p = prediction.item()
                new_row = pd.DataFrame({'anonymized_subject_id': [anonymized_subject_id], 'anonymized_session_id': [anonymized_session_id], 'predicted_score': [prediction_p]})
                df = pd.concat([df, new_row], ignore_index=True)
    else:
        print("No files found matching the pattern.")

    return df

if __name__ == "__main__":
    model_save_location = sys.argv[1]
    csv_file_name = sys.argv[2]
    dir = "/home/feczk001/shared/projects/S1067_Loes/data/Fairview-ag/"
    directory_path = os.path.join(dir, '05-training_ready')
    file_pattern = '*_RAVEL.nii.gz'
    model = load_model('resnet', model_save_location, device='cpu')
    df = make_predictions_on_folder(directory_path, file_pattern, model)
    csv_path = os.path.join(dir, csv_file_name)
    df.to_csv(csv_path, index=False)
    expected_df = pd.read_csv('/users/9/reine097/projects/loes-scoring-2/data/anon_train_scans_and_loes_training_test_non_gd.csv')
    expected_validation_df = expected_df[expected_df['validation'] == 1]
    merged_df = pd.merge(df, expected_validation_df, on=['anonymized_subject_id', 'anonymized_session_id'], how='inner')
    actual_scores = list(merged_df['loes-score'])
    predict_vals = list(merged_df['predicted_score'])
    standardized_rmse = compute_standardized_rmse(actual_scores, predict_vals)
    print(f'standardized_rmse: {standardized_rmse}')
    create_scatter_plot(actual_scores, predict_vals, '/users/9/reine097/projects/loes-scoring-2/doc/models/model21/model21.png')
