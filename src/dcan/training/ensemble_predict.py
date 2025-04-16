import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import logging
from sklearn.metrics import confusion_matrix

# Import your model and dataset classes
from dcan.models.advanced_mri_models import get_advanced_mri_model
from dcan.data_sets.dsets import LoesScoreDataset

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Ensemble prediction for ALD classification')
    parser.add_argument('--csv-input-file', required=True, help='CSV data file')
    parser.add_argument('--csv-output-file', required=True, help='CSV output file for predictions')
    parser.add_argument('--folder', required=True, help='Folder where MRIs are stored')
    parser.add_argument('--model1-path', required=True, help='Path to first model')
    parser.add_argument('--model2-path', required=True, help='Path to second model')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for inference')
    parser.add_argument('--ensemble-weights', type=str, default='0.5,0.5', 
                       help='Comma-separated weights for each model')
    parser.add_argument('--threshold', type=float, default=0.6, help='Classification threshold')
    return parser.parse_args()

def load_model(model_path, device):
    """Load a trained model from disk"""
    model = get_advanced_mri_model(model_type='resnet3d')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def create_dataloader(csv_file, folder, batch_size):
    """Create a DataLoader for inference"""
    df = pd.read_csv(csv_file)
    # Get all subjects
    subjects = list(set(df['anonymized_subject_id'].tolist()))
    
    # Create dataset and dataloader
    dataset = LoesScoreDataset(folder, subjects, df, None, is_val_set_bool=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def ensemble_predict(models, dataloader, weights, device, threshold=0.6):
    """Generate ensemble predictions"""
    all_probs = []
    all_preds = []
    all_subjects = []
    all_sessions = []
    
    with torch.no_grad():
        for batch_tup in dataloader:
            # Extract data and metadata
            input_t, _, _, subject_list, session_list = batch_tup
            input_g = input_t.to(device)
            
            # Get predictions from each model
            batch_probs = []
            for i, model in enumerate(models):
                prob = model(input_g).squeeze().cpu().numpy()
                batch_probs.append(prob * weights[i])
            
            # Compute weighted ensemble probability
            ensemble_prob = np.sum(batch_probs, axis=0)
            
            # Handle both single-item and multi-item batches
            if np.isscalar(ensemble_prob):
                # If it's a single scalar value
                ensemble_pred = float(ensemble_prob >= threshold)
                all_probs.append(float(ensemble_prob))
                all_preds.append(ensemble_pred)
            else:
                # If it's an array
                ensemble_pred = (ensemble_prob >= threshold).astype(float)
                all_probs.extend(ensemble_prob.tolist())
                all_preds.extend(ensemble_pred.tolist())
            
            # Handle subject and session lists
            if isinstance(subject_list, str):
                all_subjects.append(subject_list)
                all_sessions.append(session_list)
            else:
                all_subjects.extend(subject_list)
                all_sessions.extend(session_list)
    
    # Create result DataFrame
    results = pd.DataFrame({
        'anonymized_subject_id': all_subjects,
        'anonymized_session_id': all_sessions,
        'ensemble_probability': all_probs,
        'prediction': all_preds
    })
    
    return results

def main():
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Using device: {device}")
    
    # Parse ensemble weights
    weights = [float(w) for w in args.ensemble_weights.split(',')]
    if len(weights) != 2:
        raise ValueError("Expected 2 weights for ensemble")
    
    # Normalize weights to sum to 1
    weights = np.array(weights) / sum(weights)
    log.info(f"Using ensemble weights: {weights}")
    
    # Load models
    log.info("Loading models...")
    model1 = load_model(args.model1_path, device)
    model2 = load_model(args.model2_path, device)
    models = [model1, model2]

    # Create dataloader
    log.info("Creating dataloader...")
    dataloader = create_dataloader(args.csv_input_file, args.folder, args.batch_size)
    
    # Generate ensemble predictions
    log.info("Generating ensemble predictions...")
    results = ensemble_predict(models, dataloader, weights, device, args.threshold)
    
    # Merge with original data
    input_df = pd.read_csv(args.csv_input_file)
    merged_df = input_df.merge(
        results[['anonymized_subject_id', 'anonymized_session_id', 'ensemble_probability', 'prediction']],
        on=['anonymized_subject_id', 'anonymized_session_id'],
        how='left'
    )
    
    # Save results
    log.info(f"Saving results to {args.csv_output_file}")
    merged_df.to_csv(args.csv_output_file, index=False)
    
    # Calculate some metrics if 'has_ald' column exists
    if 'has_ald' in merged_df.columns:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        valid_rows = merged_df.dropna(subset=['prediction', 'has_ald'])
        y_true = valid_rows['has_ald'].values
        y_pred = valid_rows['prediction'].values
        y_prob = valid_rows['ensemble_probability'].values
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        log.info(f"Ensemble Metrics:")
        log.info(f"Accuracy: {accuracy:.4f}")
        log.info(f"Precision: {precision:.4f}")
        log.info(f"Recall: {recall:.4f}")
        log.info(f"F1 Score: {f1:.4f}")
        
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_prob)
            log.info(f"AUC: {auc:.4f}")

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Specificity calculation
        # True Negative / (True Negative + False Positive)
        # In confusion matrix format: cm[0,0] / (cm[0,0] + cm[0,1])
        if cm.shape == (2, 2):  # Make sure we have a 2x2 matrix
            specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
            log.info(f"Specificity: {specificity:.4f}")
            
            # You can also calculate sensitivity (same as recall) directly from confusion matrix
            sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
            log.info(f"Sensitivity: {sensitivity:.4f}")
            
            # Print confusion matrix for reference
            log.info("Confusion Matrix:")
            log.info(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
            log.info(f"FN: {cm[1,0]}, TP: {cm[1,1]}")

if __name__ == "__main__":
    main()
