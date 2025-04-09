# TODO
# Error Handling:
# If the loes_score column is missing, the script will raise a KeyError. Consider adding a check to ensure the column exists before proceeding.

# Stratification:
# Ensure loes_score has sufficient class diversity to avoid issues during stratification.

# Data Directory:
# Verify that the path /home/feczk001/shared/projects/S1067_Loes/data/Fairview-ag/ is accessible and writable for saving the output file.

# Random State:
# The random_state ensures reproducibility. Consider parameterizing it for flexibility.

# Validation of Output:
# Confirm the contents of Nascene_deID_files_train_val_test.csv align with expectations, including train_validate_test values.

import os
import pandas as pd
from sklearn.model_selection import train_test_split

from dcan.training.create_train_test_split import create_train_test_split

# Load a dataset
data_dir = 'data'
data = pd.read_csv(os.path.join(data_dir, 'anon_train_scans_and_loes_training_test_non_gd_ald.csv'))
data = data[~data['scan'].str.contains('Gd')]

# Split the dataset
X = data.drop('loes-score', axis=1)
y = data['loes-score']

X_train, X_test, y_train, y_test = create_train_test_split(X, y, test_size=0.3)
X_train['test'] = 0
X_test['test'] = 1

print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)

print("\nTraining set")
print(X_train.head())

print("\nTest set")
print(X_test.head())

train_new = X_train.join(y_train)
print("\ntrain_new")
print(train_new.head())

test_new = X_test.join(y_test)
print("\ntest_new")
print(test_new.head())

data_new = pd.concat([train_new, test_new], ignore_index=True)
data_new.to_csv(os.path.join(data_dir, 'anon_train_scans_and_loes_training_test_non_gd_ald_2.csv'), index=False)
