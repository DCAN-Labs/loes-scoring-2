import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def analyze_loes_distribution(csv_file_path, gd_filter=0):
    """
    Analyze the distribution of Loes scores in your dataset
    to understand why high scores are being underestimated.
    """
    print("="*60)
    print("LOES SCORE DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Load data
    df = pd.read_csv(csv_file_path)
    print(f"Total samples loaded: {len(df)}")
    
    # Apply Gd filter if specified
    if gd_filter == 0:
        df = df[~df['scan'].str.contains('Gd')]
        print(f"After filtering Gd scans: {len(df)}")
    
    # Basic statistics
    print(f"\nLOES SCORE STATISTICS:")
    print(f"Min score: {df['loes-score'].min()}")
    print(f"Max score: {df['loes-score'].max()}")
    print(f"Mean score: {df['loes-score'].mean():.2f}")
    print(f"Median score: {df['loes-score'].median():.2f}")
    print(f"Std deviation: {df['loes-score'].std():.2f}")
    
    # Score distribution
    print(f"\nSCORE DISTRIBUTION:")
    score_counts = Counter(df['loes-score'])
    for score in sorted(score_counts.keys()):
        count = score_counts[score]
        percentage = (count / len(df)) * 100
        print(f"Score {score:4.1f}: {count:4d} samples ({percentage:5.1f}%)")
    
    # Analyze score ranges
    print(f"\nSCORE RANGE ANALYSIS:")
    ranges = {
        'Low (0-5)': df[(df['loes-score'] >= 0) & (df['loes-score'] <= 5)],
        'Medium (5-10)': df[(df['loes-score'] > 5) & (df['loes-score'] <= 10)],
        'High (10-15)': df[(df['loes-score'] > 10) & (df['loes-score'] <= 15)],
        'Very High (>15)': df[df['loes-score'] > 15]
    }
    
    for range_name, range_df in ranges.items():
        count = len(range_df)
        percentage = (count / len(df)) * 100
        print(f"{range_name:15}: {count:4d} samples ({percentage:5.1f}%)")
    
    # Subject-level analysis
    print(f"\nSUBJECT-LEVEL ANALYSIS:")
    subject_scores = df.groupby('anonymized_subject_id')['loes-score'].agg(['mean', 'min', 'max', 'count'])
    print(f"Total unique subjects: {len(subject_scores)}")
    print(f"Avg samples per subject: {subject_scores['count'].mean():.1f}")
    print(f"Subject score range: {subject_scores['mean'].min():.1f} - {subject_scores['mean'].max():.1f}")
    
    # High score subjects
    high_score_subjects = subject_scores[subject_scores['mean'] > 10]
    print(f"Subjects with mean score >10: {len(high_score_subjects)} ({len(high_score_subjects)/len(subject_scores)*100:.1f}%)")
    
    return df, subject_scores, score_counts

def simulate_stratified_split(df, train_split=0.7, random_seed=42):
    """
    Simulate the stratified split to see the distribution
    """
    print(f"\n" + "="*60)
    print("STRATIFIED SPLIT SIMULATION")
    print("="*60)
    
    np.random.seed(random_seed)
    
    # Get subject-level scores
    subject_scores = df.groupby('anonymized_subject_id')['loes-score'].mean()
    
    try:
        # Create bins for stratification
        score_bins = pd.qcut(subject_scores, q=3, labels=['low', 'medium', 'high'], duplicates='drop')
        
        training_users = []
        validation_users = []
        
        print("STRATIFICATION BINS:")
        for bin_label in score_bins.unique():
            if pd.isna(bin_label):
                continue
            bin_subjects = score_bins[score_bins == bin_label].index.tolist()
            bin_scores = subject_scores[bin_subjects]
            
            print(f"{bin_label.upper()} bin:")
            print(f"  Score range: {bin_scores.min():.1f} - {bin_scores.max():.1f}")
            print(f"  Subjects: {len(bin_subjects)}")
            print(f"  Mean score: {bin_scores.mean():.2f}")
            
            # Split this bin
            shuffled_bin = np.random.permutation(bin_subjects)
            train_split_count = int(train_split * len(bin_subjects))
            
            bin_train = shuffled_bin[:train_split_count]
            bin_val = shuffled_bin[train_split_count:]
            
            training_users.extend(bin_train)
            validation_users.extend(bin_val)
            
            print(f"  \u2192 Train: {len(bin_train)}, Validation: {len(bin_val)}")
        
        # Analyze the resulting splits
        print(f"\nSPLIT RESULTS:")
        train_scores = subject_scores[training_users]
        val_scores = subject_scores[validation_users]
        
        print(f"Training set:")
        print(f"  Subjects: {len(training_users)}")
        print(f"  Score range: {train_scores.min():.1f} - {train_scores.max():.1f}")
        print(f"  Mean score: {train_scores.mean():.2f}")
        print(f"  High score subjects (>10): {(train_scores > 10).sum()} ({(train_scores > 10).mean()*100:.1f}%)")
        
        print(f"Validation set:")
        print(f"  Subjects: {len(validation_users)}")
        print(f"  Score range: {val_scores.min():.1f} - {val_scores.max():.1f}")
        print(f"  Mean score: {val_scores.mean():.2f}")
        print(f"  High score subjects (>10): {(val_scores > 10).sum()} ({(val_scores > 10).mean()*100:.1f}%)")
        
        return training_users, validation_users, train_scores, val_scores
        
    except Exception as e:
        print(f"Stratified split failed: {e}")
        return None, None, None, None

def analyze_training_data_balance(df, training_users):
    """
    Analyze the balance of training data across score ranges
    """
    print(f"\n" + "="*60)
    print("TRAINING DATA BALANCE ANALYSIS")
    print("="*60)
    
    # Get training data
    train_df = df[df['anonymized_subject_id'].isin(training_users)]
    
    print(f"Training samples: {len(train_df)}")
    
    # Analyze score distribution in training data
    train_score_counts = Counter(train_df['loes-score'])
    print(f"\nTRAINING SCORE DISTRIBUTION:")
    
    total_train = len(train_df)
    for score in sorted(train_score_counts.keys()):
        count = train_score_counts[score]
        percentage = (count / total_train) * 100
        print(f"Score {score:4.1f}: {count:4d} samples ({percentage:5.1f}%)")
    
    # Score range analysis for training
    print(f"\nTRAINING SCORE RANGES:")
    ranges = {
        'Low (0-5)': train_df[(train_df['loes-score'] >= 0) & (train_df['loes-score'] <= 5)],
        'Medium (5-10)': train_df[(train_df['loes-score'] > 5) & (train_df['loes-score'] <= 10)],
        'High (10-15)': train_df[(train_df['loes-score'] > 10) & (train_df['loes-score'] <= 15)],
        'Very High (>15)': train_df[train_df['loes-score'] > 15]
    }
    
    for range_name, range_df in ranges.items():
        count = len(range_df)
        percentage = (count / total_train) * 100
        print(f"{range_name:15}: {count:4d} samples ({percentage:5.1f}%)")
    
    # Check for severe imbalance
    high_score_ratio = len(ranges['High (10-15)']) + len(ranges['Very High (>15)'])
    high_score_percentage = (high_score_ratio / total_train) * 100
    
    print(f"\n\u26a0\ufe0f  HIGH SCORE REPRESENTATION:")
    print(f"High scores (>10): {high_score_ratio} samples ({high_score_percentage:.1f}%)")
    
    if high_score_percentage < 10:
        print("\U0001f6a8 WARNING: Very low representation of high scores in training data!")
        print("   This likely explains why the model underestimates high scores.")
    elif high_score_percentage < 20:
        print("\u26a0\ufe0f  CAUTION: Low representation of high scores in training data.")
    else:
        print("\u2705 Reasonable representation of high scores in training data.")
    
    return train_df, high_score_percentage

def recommend_solutions(high_score_percentage, score_counts):
    """
    Recommend solutions based on the data analysis
    """
    print(f"\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    if high_score_percentage < 10:
        print("\U0001f3af PRIORITY SOLUTIONS:")
        print("1. USE WEIGHTED LOSS:")
        print("   Add --use-weighted-loss to your training command")
        print("   This will increase penalty for misclassifying rare high scores")
        
        print("\n2. CHANGE SPLIT STRATEGY:")
        print("   Use --split-strategy random instead of stratified")
        print("   Stratified split might be isolating high scores in validation")
        
        print("\n3. INCREASE TRAINING DATA RATIO:")
        print("   Use --train-split 0.8 instead of 0.7")
        print("   More training data = better high score representation")
        
        print("\n4. FOCAL LOSS (Advanced):")
        print("   Consider implementing focal loss for extreme class imbalance")
        
    elif high_score_percentage < 20:
        print("\U0001f3af RECOMMENDED SOLUTIONS:")
        print("1. Enable weighted loss: --use-weighted-loss")
        print("2. Try random split: --split-strategy random")
        print("3. Consider data augmentation for high-score cases")
    
    else:
        print("\U0001f3af ALTERNATIVE SOLUTIONS:")
        print("1. Check model capacity - might need larger model")
        print("2. Adjust learning rate or scheduler")
        print("3. Increase training epochs")
    
    # Analyze extreme imbalance
    max_count = max(score_counts.values())
    min_count = min(score_counts.values())
    imbalance_ratio = max_count / min_count
    
    print(f"\n\U0001f4ca CLASS IMBALANCE RATIO: {imbalance_ratio:.1f}:1")
    if imbalance_ratio > 50:
        print("\U0001f6a8 SEVERE IMBALANCE: Weighted loss is essential!")
    elif imbalance_ratio > 10:
        print("\u26a0\ufe0f  MODERATE IMBALANCE: Weighted loss recommended")

# Usage example
if __name__ == "__main__":
    # Replace with your actual CSV file path
    csv_file = "/users/9/reine097/projects/loes-scoring-2/data/regression.csv"
    
    # Analyze the data
    df, subject_scores, score_counts = analyze_loes_distribution(csv_file, gd_filter=0)
    
    # Simulate your current stratified split
    training_users, validation_users, train_scores, val_scores = simulate_stratified_split(
        df, train_split=0.7, random_seed=42
    )
    
    if training_users is not None:
        # Analyze training data balance
        train_df, high_score_percentage = analyze_training_data_balance(df, training_users)
        
        # Get recommendations
        recommend_solutions(high_score_percentage, score_counts)
    
    print(f"\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    