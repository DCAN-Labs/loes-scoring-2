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

# Load a dataset
data_dir = '/home/feczk001/shared/projects/S1067_Loes/data/Fairview-ag/'
data = pd.read_csv(os.path.join(data_dir, 'Nascene_deID_files.csv'))
data['train_validate_test'] = 0

# Split the dataset
X = data.drop('loes_score', axis=1)
y = data['loes_score']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

print("Training set shape:", X_train.shape, y_train.shape)

# Split the temporary set into validation and test sets
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
X_val['train_validate_test'] = 1
X_test['train_validate_test'] = 2

print("Validation set shape:", X_val.shape, y_val.shape)
print("Testing set shape:", X_test.shape, y_test.shape)

print("\nTraining set")
print(X_train.head())

print("\nValidation set")
print(X_val.head())

print("\nTest set")
print(X_test.head())

train_new = X_train.join(y_train)
print("\ntrain_new")
print(train_new.head())

val_new = X_val.join(y_val)
print("\nval_new")
print(val_new.head())

test_new = X_test.join(y_test)
print("\ntest_new")
print(test_new.head())

data_new = pd.concat([train_new, val_new, test_new], ignore_index=True)
data_new.to_csv(os.path.join(data_dir, 'Nascene_deID_files_train_val_test.csv'), index=False)
