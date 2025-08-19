"""LOES Score Prediction Module

This module provides functionality for predicting LOES (Loes) scores from brain MRI scans
using deep learning models (ResNet or AlexNet3D). It handles model loading, image preprocessing,
inference, and validation against known scores.

Key Features:
    - Load pre-trained ResNet or AlexNet3D models for LOES score prediction
    - Process NIfTI brain MRI files with z-normalization
    - Batch predictions on directories of MRI scans
    - Validation and comparison with ground truth scores
    - Scatter plot visualization of predictions vs actual scores
    - RMSE and correlation coefficient computation

Usage:
    python make_predictions.py <model_file> <output_csv> <nifti_dir> [options]
    
Example:
    python make_predictions.py model.pt predictions.csv /data/mri_scans \
        --model_type resnet --device gpu --validation_csv_file_path validation.csv
"""

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
import argparse

from dcan.inference.models import AlexNet3D
from dcan.models.ResNet import get_resnet_model

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)


def load_model(model_name, model_save_location, device='cpu'):
    """Load a pre-trained LOES scoring model.
    
    Args:
        model_name (str): Model architecture type ('resnet' or 'alexnet')
        model_save_location (str): Path to saved model checkpoint (.pt file)
        device (str): Device to load model on ('cpu' or 'cuda')
    
    Returns:
        torch.nn.Module: Loaded model in evaluation mode
    
    Raises:
        FileNotFoundError: If model file doesn't exist
        PermissionError: If model file is not readable
        ValueError: If model file has invalid format
        RuntimeError: If model architecture doesn't match checkpoint
    """
    # Check if model file exists
    if not os.path.exists(model_save_location):
        raise FileNotFoundError(f"Model file not found at: {model_save_location}")
    
    # Check if the file is readable
    if not os.access(model_save_location, os.R_OK):
        raise PermissionError(f"Cannot read model file at: {model_save_location}")
    
    # Initialize model based on type
    if model_name.lower() == 'resnet':
        model = get_resnet_model()
        log.info("Using ResNet")
    else:
        model = AlexNet3D(4608)
        log.info("Using AlexNet3D")
    
    model.to(device)
    
    # Load model with error handling
    try:
        state_dict = torch.load(model_save_location, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        log.info(f"Successfully loaded model from {model_save_location}")
    except torch.serialization.pickle.UnpicklingError as e:
        raise ValueError(f"Invalid model file format at {model_save_location}: {e}")
    except RuntimeError as e:
        raise RuntimeError(f"Model architecture mismatch or corrupted file at {model_save_location}: {e}")
    except Exception as e:
        raise Exception(f"Failed to load model from {model_save_location}: {e}")
    
    model.eval()
    return model
    

def predict(row, data_folder):
    """Generate input tensor for a single subject/session from CSV row.
    
    Args:
        row (pd.Series): DataFrame row with subject and session IDs
        data_folder (str): Directory containing preprocessed NIfTI files
    
    Returns:
        torch.Tensor: Preprocessed image tensor ready for model input
    """
    subject = row['anonymized_subject_id']
    session = row['anonymized_session_id']
    mprage_path = f'{data_folder}/{subject}_{session}_space-MNI_brain_mprage_RAVEL.nii.gz'
    mprage_image_tensor = get_image_tensor(mprage_path)
    value = mprage_image_tensor.unsqueeze(0)

    return value

def z_normalize(image, mask=None):
    """Apply z-normalization (standardization) to image data.
    
    Args:
        image (np.ndarray): Input image array
        mask (np.ndarray, optional): Binary mask for selective normalization
    
    Returns:
        np.ndarray: Z-normalized image with mean=0, std=1
    """
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
    """Load and preprocess a NIfTI MRI scan for model input.
    
    Args:
        mprage_path (str): Path to NIfTI file (.nii.gz format)
    
    Returns:
        torch.Tensor: Preprocessed 4D tensor (1, C, H, W, D) with z-normalization applied
    """
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
    """Calculate Root Mean Square Error between predictions and actual values.
    
    Args:
        predictions (list): Predicted LOES scores
        actuals (list): Actual/ground truth LOES scores
    
    Returns:
        float: RMSE value
    """
    predictions_tensor = torch.tensor(predictions, dtype=torch.float32)
    actuals_tensor = torch.tensor(actuals, dtype=torch.float32)
    mse = F.mse_loss(predictions_tensor, actuals_tensor)
    return torch.sqrt(mse).item()


def get_validation_info(model_type, model_save_location, input_csv_location, val_subjects, data_folder):
    """Generate predictions for validation subjects.
    
    Args:
        model_type (str): Model architecture ('resnet' or 'alexnet')
        model_save_location (str): Path to model checkpoint
        input_csv_location (str): CSV with subject metadata
        val_subjects (list): List of validation subject IDs
        data_folder (str): Directory with preprocessed MRI files
    
    Returns:
        tuple: (subjects, sessions, actual_scores, predicted_scores)
    """
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
    """Compute RMSE normalized by standard deviation of actual scores.
    
    Args:
        actual_scores (list): Ground truth LOES scores
        predict_vals (list): Predicted LOES scores
    
    Returns:
        float: Standardized RMSE (RMSE / σ_actual)
    """
    rmse = compute_rmse(predict_vals, actual_scores)
    sigma = statistics.stdev(actual_scores)
    standardized_rmse = rmse / sigma
    
    return standardized_rmse


def create_correlation_coefficient(actual_vals, predicted_vals):
    """Calculate Pearson correlation coefficient between actual and predicted values.
    
    Args:
        actual_vals (list): Actual LOES scores
        predicted_vals (list): Predicted LOES scores
    
    Returns:
        float: Pearson correlation coefficient (r value)
    """
    x = np.array(actual_vals)
    y = np.array(predicted_vals)

    correlation_matrix = np.corrcoef(x, y)
    correlation_coefficient = correlation_matrix[0, 1]

    return correlation_coefficient


def create_scatter_plot(actual_vals, predicted_vals, output_file):
    """Generate scatter plot comparing predicted vs actual LOES scores.
    
    Creates a scatter plot with:
    - Points colored by prediction error magnitude
    - Perfect prediction diagonal line
    - Equal aspect ratio for fair comparison
    
    Args:
        actual_vals (list): Ground truth LOES scores
        predicted_vals (list): Model predicted LOES scores
        output_file (str): Path to save plot image
    """
    _, ax = plt.subplots(figsize=(8, 6))
    
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
    """Match predicted value to subject/session combination.
    
    Args:
        row (pd.Series): DataFrame row with subject and session IDs
        subjects (list): List of subject IDs
        sessions (list): List of session IDs
        predict_vals (list): Corresponding predicted values
    
    Returns:
        float: Predicted value for the subject/session, or NaN if not found
    """
    zipped_data = zip(subjects, sessions, predict_vals)

    for subject, session, predict_val in zipped_data:
        if row['anonymized_subject_id'] == subject and row['anonymized_session_id'] == session:
            return predict_val
    return np.nan

def add_predicted_values(subjects, sessions, predict_vals, input_csv_location):
    """Add predicted LOES scores to existing CSV data.
    
    Args:
        subjects (list): Subject IDs with predictions
        sessions (list): Session IDs with predictions
        predict_vals (list): Predicted LOES scores
        input_csv_location (str): Path to input CSV file
    
    Returns:
        pd.DataFrame: Input data with added 'predicted_loes_score' column
    """
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
    """Generate LOES predictions for all matching MRI files in a directory.
    
    Processes all NIfTI files matching the pattern, extracting subject/session IDs
    from filenames (expected format: <subject>_<session>_*.nii.gz).
    
    Args:
        directory_path (str): Directory containing MRI files
        file_pattern (str): Glob pattern for MRI files (e.g., '*_RAVEL.nii.gz')
        model (torch.nn.Module): Loaded model for prediction
    
    Returns:
        pd.DataFrame: DataFrame with columns: anonymized_subject_id, 
                     anonymized_session_id, predicted_score
    """
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
    parser = argparse.ArgumentParser(description='Make predictions on input files.')
    parser.add_argument('model_file_path', help='Path to the model file')
    parser.add_argument('input_csv_file_path', help='Path to the CSV file')
    parser.add_argument('nifti_directory_path', help='Path to folder containing NIFTI files')
    parser.add_argument('--file_pattern', default='*_RAVEL.nii.gz', help='The pattern of the NIFTI files, such as _RAVEL.nii.gz')
    parser.add_argument('--device', default='gpu', choices=['gpu', 'cpu'], help='Choose the device (gpu or cpu)')
    parser.add_argument('--model_type', default='resnet', choices=['resnet', 'alexnet'], help='Choose the model architecture (resnet or alexnet)')
    parser.add_argument('--validation_csv_file_path', help="For validation of known Loes scores, the path to the CSV containing those scores.")
    parser.add_argument('--scatter_plot_file_path', help="File path to output scatter plot")
    args = parser.parse_args()
    model_save_location = sys.argv[1]
    csv_file_name = sys.argv[2]
    directory_path = sys.argv[3]
    file_pattern = args.file_pattern
    model = load_model(args.model_type, model_save_location, device=args.device)
    df = make_predictions_on_folder(directory_path, file_pattern, model)
    df.to_csv(csv_file_name, index=False)
    if args.validation_csv_file_path:
        expected_df = pd.read_csv(args.validation_csv_file_path)
        expected_df = pd.read_csv()
        expected_validation_df = expected_df[expected_df['validation'] == 1]
        merged_df = pd.merge(df, expected_validation_df, on=['anonymized_subject_id', 'anonymized_session_id'], how='inner')
        actual_scores = list(merged_df['loes-score'])
        predict_vals = list(merged_df['predicted_score'])
        standardized_rmse = compute_standardized_rmse(actual_scores, predict_vals)
        print(f'standardized_rmse: {standardized_rmse}')
        create_scatter_plot(actual_scores, predict_vals, args.scatter_plot_file_path)
