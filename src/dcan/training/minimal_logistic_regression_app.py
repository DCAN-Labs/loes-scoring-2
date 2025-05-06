import torch
import pandas as pd
import numpy as np

from dcan.training.logistic_regression import Config


def main():
    """
    Entry point for the application.
    """
    try:
        # Create a simplified version of LogisticRegressionApp
        # that just loads the data and sets up cross-validation
        app = MinimalLogisticRegressionApp()
        app.run_cross_validation(k=5)
    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()

class MinimalLogisticRegressionApp:
    """
    A minimal version of LogisticRegressionApp that focuses just on
    the cross-validation functionality.
    """
    def __init__(self):
        # Parse arguments
        self.config = Config().parse_args()
        
        # Set up basic attributes
        self.use_cuda = torch.cuda.is_available() if not self.config.DEBUG else False
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        
        # Load data
        print("Loading data...")
        self.input_df = pd.read_csv(self.config.csv_input_file)
        self.output_df = self.input_df.copy()
        self.output_df["prediction"] = np.nan
        print(f"Data loaded: {len(self.input_df)} rows")
    
    def run_cross_validation(self, k=5):
        """
        Run a simplified cross-validation process.
        """
        print(f"Starting {k}-fold cross-validation")
        
        # Get all unique subjects
        all_subjects = list(set(self.input_df['anonymized_subject_id'].tolist()))
        print(f"Total subjects: {len(all_subjects)}")
        
        # Split into folds
        import random
        random.shuffle(all_subjects)
        
        # Create folds
        folds = [[] for _ in range(k)]
        for i, subject in enumerate(all_subjects):
            folds[i % k].append(subject)
        
        # Report results
        for i, fold in enumerate(folds):
            print(f"Fold {i+1}: {len(fold)} subjects")
        
        print("Cross-validation complete")
        