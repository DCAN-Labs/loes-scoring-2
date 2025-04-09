import math
import random
import os
import pandas as pd
import logging
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
log = logging.getLogger(__name__)

# Add random seed for reproducibility
random.seed(42)

def every_nth(lst, n):
    """Returns a new list containing every nth element of the input list."""
    if not lst:
        return []
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    return lst[n-1::n]

def split_by_ald_present(df_in, ald_threshold=0.1):
    """
    Split the DataFrame into two based on ALD presence determined by loes-score.
    
    Args:
        df_in (pd.DataFrame): Input DataFrame with 'anonymized_subject_id' and 'loes-score' columns
        ald_threshold (float): Threshold for determining ALD presence
        
    Returns:
        tuple: (no_ald_df, ald_df) - DataFrames with subjects without and with ALD
        
    Raises:
        ValueError: If required columns are missing or if no subjects meet criteria
    """
    # Validate input DataFrame
    required_columns = ['anonymized_subject_id', 'loes-score']
    for col in required_columns:
        if col not in df_in.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")
    
    # Check for empty DataFrame
    if df_in.empty:
        raise ValueError("Input DataFrame is empty")
    
    # Check for missing values
    missing_values = df_in[required_columns].isnull().sum()
    if missing_values.sum() > 0:
        log.warning(f"Missing values detected: {missing_values}")
    
    try:
        # More efficient implementation using pandas groupby
        grouped = df_in.groupby('anonymized_subject_id')['loes-score'].max()
        non_ald_subjects = grouped[grouped < ald_threshold].index.tolist()
        
        if not non_ald_subjects:
            log.warning("No subjects classified as non-ALD")
            
        mask = df_in['anonymized_subject_id'].isin(non_ald_subjects)
        no_ald_df = df_in[mask]
        ald_df = df_in[~mask]
        
        # Validate split result
        log.info(f"Split result: {len(no_ald_df)} rows without ALD, {len(ald_df)} rows with ALD")
        log.info(f"Split result: {len(set(no_ald_df['anonymized_subject_id']))} subjects without ALD, " 
                 f"{len(set(ald_df['anonymized_subject_id']))} subjects with ALD")
        
        if no_ald_df.empty or ald_df.empty:
            log.warning("One of the split DataFrames is empty")
            
        return no_ald_df, ald_df
        
    except Exception as e:
        log.error(f"Error splitting data: {e}")
        raise

def split_data(df, test_val_ratio=0.2, test_ratio=0.5):
    """
    Split subjects into training, validation, and test sets.
    
    Args:
        df (pd.DataFrame): Input DataFrame with 'anonymized_subject_id'
        test_val_ratio (float): Ratio of subjects for test+validation (default: 0.2)
        test_ratio (float): Ratio of test subjects within test+validation (default: 0.5)
        
    Returns:
        tuple: (training_subjects, val_subjects, test_subjects) - Lists of subject IDs
        
    Raises:
        ValueError: If DataFrame is invalid or if splitting results in empty sets
    """
    if 'anonymized_subject_id' not in df.columns:
        raise ValueError("Column 'anonymized_subject_id' not found in DataFrame")
        
    if df.empty:
        raise ValueError("Input DataFrame is empty")
    
    try:
        df_subjects = list(set(df['anonymized_subject_id']))
        if not df_subjects:
            raise ValueError("No subjects found in DataFrame")
            
        # Validate parameters
        if not 0 < test_val_ratio < 1:
            raise ValueError("test_val_ratio must be between 0 and 1")
        if not 0 < test_ratio < 1:
            raise ValueError("test_ratio must be between 0 and 1")
        
        random.shuffle(df_subjects)
        
        # Calculate nth value based on test_val_ratio
        n_value = int(round(1 / test_val_ratio))
        test_or_val_subjects = every_nth(df_subjects, n_value)
        
        # Use test_ratio to split test and validation
        test_subjects = every_nth(test_or_val_subjects, int(1 / test_ratio))
        val_subjects = [subject for subject in test_or_val_subjects if subject not in test_subjects]
        training_subjects = [subject for subject in df_subjects if subject not in test_or_val_subjects]
        
        # Validate split results
        if not training_subjects or not val_subjects or not test_subjects:
            log.warning("One of the split sets is empty")
            
        log.info(f"Split result: {len(training_subjects)} training, "
                 f"{len(val_subjects)} validation, "
                 f"{len(test_subjects)} test subjects")
                 
        return training_subjects, val_subjects, test_subjects
        
    except Exception as e:
        log.error(f"Error splitting data: {e}")
        raise

def validate_output_dataframe(df):
    """
    Validate the output DataFrame to ensure proper splitting.
    
    Args:
        df (pd.DataFrame): DataFrame with training, validation, and test flags
        
    Returns:
        bool: True if validation passes
    """
    # Check that each row is assigned to exactly one set
    row_sum = df[['training', 'validation', 'test']].sum(axis=1)
    
    if not all(row_sum == 1):
        multi_assigned = (row_sum > 1).sum()
        unassigned = (row_sum == 0).sum()
        log.error(f"{multi_assigned} rows assigned to multiple sets, {unassigned} rows unassigned")
        return False
        
    # Check reasonable split sizes (approximately 70/15/15 or 80/10/10)
    train_pct = df['training'].mean() * 100
    val_pct = df['validation'].mean() * 100
    test_pct = df['test'].mean() * 100
    
    log.info(f"Split percentages: {train_pct:.1f}% training, "
             f"{val_pct:.1f}% validation, {test_pct:.1f}% test")
             
    if train_pct < 60 or val_pct < 5 or test_pct < 5:
        log.warning("Split percentages are outside expected ranges")
        
    return True

def print_data_distribution_summary(df, id_column='anonymized_subject_id'):
    """
    Print a summary of data distribution after splitting.
    
    Args:
        df (pd.DataFrame): DataFrame with training, validation, and test flags
        id_column (str): Column name containing subject IDs
    """
    log.info("\n" + "="*50)
    log.info("DATA DISTRIBUTION SUMMARY")
    log.info("="*50)
    
    # Overall statistics
    total_rows = len(df)
    total_subjects = df[id_column].nunique()
    log.info(f"Total: {total_rows} rows, {total_subjects} unique subjects")
    
    # Split statistics
    train_df = df[df['training'] == 1]
    val_df = df[df['validation'] == 1]
    test_df = df[df['test'] == 1]
    
    log.info("\nRow distribution:")
    log.info(f"Training:   {len(train_df)} rows ({len(train_df)/total_rows*100:.1f}%)")
    log.info(f"Validation: {len(val_df)} rows ({len(val_df)/total_rows*100:.1f}%)")
    log.info(f"Test:       {len(test_df)} rows ({len(test_df)/total_rows*100:.1f}%)")
    
    log.info("\nSubject distribution:")
    train_subjects = train_df[id_column].nunique()
    val_subjects = val_df[id_column].nunique()
    test_subjects = test_df[id_column].nunique()
    log.info(f"Training:   {train_subjects} subjects ({train_subjects/total_subjects*100:.1f}%)")
    log.info(f"Validation: {val_subjects} subjects ({val_subjects/total_subjects*100:.1f}%)")
    log.info(f"Test:       {test_subjects} subjects ({test_subjects/total_subjects*100:.1f}%)")
    
    # Check for Loes score distribution if available
    if 'loes-score' in df.columns:
        log.info("\nLoes score statistics:")
        log.info(f"Overall mean: {df['loes-score'].mean():.2f} (std: {df['loes-score'].std():.2f})")
        log.info(f"Training mean: {train_df['loes-score'].mean():.2f} (std: {train_df['loes-score'].std():.2f})")
        log.info(f"Validation mean: {val_df['loes-score'].mean():.2f} (std: {val_df['loes-score'].std():.2f})")
        log.info(f"Test mean: {test_df['loes-score'].mean():.2f} (std: {test_df['loes-score'].std():.2f})")
        
        # Count of ALD/non-ALD in each split (using the same threshold)
        ald_threshold = 0.1  # Should match the threshold used in split_by_ald_present
        
        # Get count of subjects with any scan having loes >= threshold
        def count_ald_subjects(df_subset):
            return df_subset.groupby(id_column)['loes-score'].max().ge(ald_threshold).sum()
            
        train_ald = count_ald_subjects(train_df)
        val_ald = count_ald_subjects(val_df)
        test_ald = count_ald_subjects(test_df)
        
        log.info("\nALD distribution (subjects):")
        log.info(f"Training:   {train_ald} ALD, {train_subjects - train_ald} non-ALD")
        log.info(f"Validation: {val_ald} ALD, {val_subjects - val_ald} non-ALD")
        log.info(f"Test:       {test_ald} ALD, {test_subjects - test_ald} non-ALD")
    
    log.info("="*50)

def main(csv_data_file_in=None, csv_data_file_out=None, ald_threshold=0.1, 
         test_val_ratio=0.2, test_ratio=0.5):
    """
    Main function to split data into training, validation and test sets.
    
    Args:
        csv_data_file_in (str): Path to input CSV file
        csv_data_file_out (str): Path to output CSV file
        ald_threshold (float): Threshold for determining ALD presence
        test_val_ratio (float): Ratio of subjects for test+validation
        test_ratio (float): Ratio of test subjects within test+validation
    """
    try:
        # Use default paths if not provided
        if csv_data_file_in is None:
            csv_data_file_in = "/users/9/reine097/projects/loes-scoring-2/data/anon_train_scans_and_loes_training_test_non_gd.csv"
        
        if csv_data_file_out is None:
            csv_data_file_out = "/users/9/reine097/projects/loes-scoring-2/data/anon_train_scans_and_loes_training_test_non_gd_logistic_regression.csv"
        
        # Log parameter values
        log.info(f"Parameters: ald_threshold={ald_threshold}, test_val_ratio={test_val_ratio}, test_ratio={test_ratio}")
        
        # Validate input file exists
        if not os.path.exists(csv_data_file_in):
            raise FileNotFoundError(f"Input file not found: {csv_data_file_in}")
            
        # Validate output directory exists
        output_dir = os.path.dirname(csv_data_file_out)
        if not os.path.exists(output_dir):
            log.warning(f"Output directory does not exist: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
            log.info(f"Created output directory: {output_dir}")
        
        log.info(f"Loading data from {csv_data_file_in}")
        in_df = pd.read_csv(csv_data_file_in)
        log.info(f"Loaded {len(in_df)} rows with {in_df['anonymized_subject_id'].nunique()} unique subjects")
        
        out_df = in_df.copy()
        
        no_ald_df, ald_df = split_by_ald_present(in_df, ald_threshold=ald_threshold)
        no_ald_training_subjects, no_ald_val_subjects, no_ald_test_subjects = split_data(
            no_ald_df, test_val_ratio=test_val_ratio, test_ratio=test_ratio)
        ald_training_subjects, ald_val_subjects, ald_test_subjects = split_data(
            ald_df, test_val_ratio=test_val_ratio, test_ratio=test_ratio)
        
        def is_training(row):
            if row['anonymized_subject_id'] in no_ald_training_subjects + ald_training_subjects:
                return 1
            else:
                return 0
                
        def is_validation(row):
            if row['anonymized_subject_id'] in no_ald_val_subjects + ald_val_subjects:
                return 1
            else:
                return 0
                
        def is_test(row):
            if row['anonymized_subject_id'] in no_ald_test_subjects + ald_test_subjects:
                return 1
            else:
                return 0
                
        log.info("Assigning split flags to output DataFrame")
        out_df['training'] = out_df.apply(is_training, axis=1)
        out_df['validation'] = out_df.apply(is_validation, axis=1)
        out_df['test'] = out_df.apply(is_test, axis=1)
        
        # Validate output DataFrame
        validate_output_dataframe(out_df)
        
        # Print detailed summary of the data distribution
        print_data_distribution_summary(out_df)
        
        # Save output
        log.info(f"Saving results to {csv_data_file_out}")
        out_df.to_csv(csv_data_file_out, index=False)
        log.info("Data splitting completed successfully")
        
    except Exception as e:
        log.error(f"ERROR: {e}")
        raise

if __name__ == '__main__':
    # You can call main with different parameters here
    main('data/regression_data.csv', 'data/logistic_regression_data.csv')