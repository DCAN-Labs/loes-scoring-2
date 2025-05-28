import logging
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)

class ROCPlotter:
    def __init__(self, save_dir: str = None, time_str: str = None):
        self.save_dir = save_dir
        self.time_str = time_str

    def plot_single_roc(self, y_true, y_prob, save_path=None, fold_idx=None, threshold=0.5, **kwargs):
        """
        Plot ROC curve for given predictions
        
        Args:
            y_true: True labels (numpy array)
            y_prob: Predicted probabilities (numpy array)
            save_path: Path to save the plot
            fold_idx: Fold index for cross-validation (optional)
            threshold: Classification threshold for metrics calculation
        """
        log.info("Plotting ROC curve...")
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # Calculate partial AUC (pAUC) at FPR <= 0.1
        pauc_fpr_limit = 0.1
        limited_fpr = fpr[fpr <= pauc_fpr_limit]
        limited_tpr = tpr[:len(limited_fpr)]
        
        if len(limited_fpr) > 1:
            pAUC = auc(limited_fpr, limited_tpr) / pauc_fpr_limit
        else:
            pAUC = 0.0
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        
        # Plot ROC curve
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        
        # Mark current threshold point
        current_threshold_idx = np.argmin(np.abs(thresholds - threshold))
        plt.plot(fpr[current_threshold_idx], tpr[current_threshold_idx], 
                'ro', markersize=10, 
                label=f'Current threshold ({threshold:.3f})')
        
        # Find and mark optimal threshold (Youden's J statistic)
        youden_j = tpr - fpr
        optimal_idx = np.argmax(youden_j)
        optimal_threshold = thresholds[optimal_idx]
        plt.plot(fpr[optimal_idx], tpr[optimal_idx], 
                'go', markersize=10, 
                label=f'Optimal threshold ({optimal_threshold:.3f})')
        
        # Customize plot
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        
        if fold_idx is not None:
            plt.title(f'ROC Curve - Fold {fold_idx + 1}')
        else:
            plt.title('Receiver Operating Characteristic (ROC) Curve')
        
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Add metrics text box
        metrics_text = f'Metrics at current threshold ({threshold:.3f}):\n'
        
        # Calculate metrics at current threshold
        y_pred = (y_prob >= threshold).astype(int)
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Calculate specificity
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            specificity = 0.0
        
        metrics_text += f'Accuracy: {accuracy:.3f}\n'
        metrics_text += f'Precision: {precision:.3f}\n'
        metrics_text += f'Recall: {recall:.3f}\n'
        metrics_text += f'Specificity: {specificity:.3f}\n'
        metrics_text += f'F1 Score: {f1:.3f}'
        
        # Add text box
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.55, 0.15, metrics_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            if self.save_dir:
                base_path = os.path.splitext(self.save_dir)[0]
                if fold_idx is not None:
                    save_path = f"{base_path}_roc_fold{fold_idx+1}.png"
                else:
                    save_path = f"{base_path}_roc.png"
            else:
                save_path = f"roc_curve_{self.time_str or 'default'}.png"
        
        # Create directory if needed
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        log.info(f"ROC curve saved to {save_path}")
        
        # Also save as PDF for publication quality
        pdf_path = save_path.rsplit('.', 1)[0] + '.pdf'
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        log.info(f"ROC curve saved to {pdf_path}")
        
        plt.close()
        
        return roc_auc, pAUC
    
 
    def plot_combined_cv_roc(self, fold_roc_data, mean_fpr, k=5):
        """
        Plot combined ROC curve showing mean performance, random classifier, and optimal threshold.
        """
        plt.figure(figsize=(10, 8))

        # Check if we have any valid folds
        if not fold_roc_data['original_fprs']:
            log.error("No valid folds with ROC data to plot")
            return
        
        # Calculate mean TPR across folds using interpolated data
        if fold_roc_data['interp_tprs']:
            mean_tpr = np.mean(fold_roc_data['interp_tprs'], axis=0)
            mean_tpr[-1] = 1.0  # Ensure it ends at (1,1)
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(fold_roc_data['aucs'])
            
            # Plot the mean ROC curve (blue line)
            plt.plot(mean_fpr, mean_tpr, color='blue', lw=3,
                    label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
            
            # Calculate and plot optimal threshold point on mean curve
            # Use Youden's J statistic to find optimal threshold
            youden_j = mean_tpr - mean_fpr
            optimal_idx = np.argmax(youden_j)
            optimal_fpr = mean_fpr[optimal_idx]
            optimal_tpr = mean_tpr[optimal_idx]
        
            # Calculate what threshold this corresponds to by using overall data
            if fold_roc_data['y_true_all'] and fold_roc_data['y_prob_all']:
                overall_fpr, overall_tpr, overall_thresholds = roc_curve(
                    fold_roc_data['y_true_all'], 
                    fold_roc_data['y_prob_all']
                )
                # Find closest point on overall curve to our optimal point
                distances = np.sqrt((overall_fpr - optimal_fpr)**2 + (overall_tpr - optimal_tpr)**2)
                closest_idx = np.argmin(distances)
                optimal_threshold = overall_thresholds[closest_idx]
                
                # Plot the optimal threshold point
                plt.plot(optimal_fpr, optimal_tpr, 'ro', markersize=10,
                        label=f'Optimal Threshold ({optimal_threshold:.3f})', zorder=5)
            else:
                # Fallback: just show the optimal point without threshold value
                plt.plot(optimal_fpr, optimal_tpr, 'ro', markersize=10,
                        label='Optimal Point (Youden\'s J)', zorder=5)

        # Plot diagonal (random classifier - dotted line)
        plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.8, label='Random Classifier')
        
        # Formatting
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title(f'CA-MEDS Performance', fontsize=16)
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)

        # Save the plot
        if self.save_dir:
            if os.path.isdir(self.save_dir):
                plot_path = os.path.join(self.save_dir, f"combined_roc_cv_{self.time_str}.png")
            else:
                base_path = os.path.splitext(self.save_dir)[0]
                plot_path = f"{base_path}_combined_roc_cv.png"
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            
            log.info(f"Attempting to save plot to: {plot_path}")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            
            # Verify the file was created
            if os.path.exists(plot_path):
                log.info(f"\u2713 Combined ROC curve successfully saved to {plot_path}")
            else:
                log.error(f"\u2717 Failed to save plot to {plot_path}")
        else:
            log.warning("No plot location configured, using fallback")
            plot_path = f"combined_roc_cv_{self.time_str or 'default'}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            log.info(f"Combined ROC curve saved to {plot_path}")
        
        plt.close()  # Close the figure to free memory
        