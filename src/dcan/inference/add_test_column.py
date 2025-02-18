import pandas as pd

val_subjects = ['subject-05', 'subject-50', 'subject-35', 'subject-40', 'subject-45', 'subject-62', 'subject-35', 'subject-50', 'subject-55', 'subject-62', 'subject-00', 'subject-30', 'subject-50', 'subject-35', 'subject-05', 'subject-00', 'subject-05', 'subject-55', 'subject-05', 'subject-25', 'subject-55', 'subject-30', 'subject-50', 'subject-55', 'subject-35', 'subject-55', 'subject-50', 'subject-62', 'subject-67', 'subject-15', 'subject-20', 'subject-25', 'subject-50', 'subject-15', 'subject-45', 'subject-45', 'subject-00', 'subject-55', 'subject-62', 'subject-62', 'subject-55', 'subject-67', 'subject-05', 'subject-45', 'subject-00', 'subject-55', 'subject-10', 'subject-45', 'subject-45', 'subject-45', 'subject-35', 'subject-40', 'subject-05', 'subject-25', 'subject-50', 'subject-30', 'subject-62', 'subject-50', 'subject-55', 'subject-50', 'subject-05', 'subject-35', 'subject-35', 'subject-10', 'subject-15', 'subject-20', 'subject-45', 'subject-45', 'subject-50', 'subject-45', 'subject-45', 'subject-62', 'subject-67', 'subject-45', 'subject-50', 'subject-30', 'subject-40', 'subject-62', 'subject-00', 'subject-05', 'subject-50', 'subject-67', 'subject-05', 'subject-10', 'subject-55', 'subject-45', 'subject-50', 'subject-10', 'subject-40', 'subject-15', 'subject-05', 'subject-45', 'subject-35', 'subject-55', 'subject-15']

def is_validation_row(row):
    if row['anonymized_subject_id'] in val_subjects and 'Gd' not in row['scan']:
        return 1
    else:
        return 0

def is_training_row(row):
    if row['anonymized_subject_id'] not in val_subjects and 'Gd' not in row['scan']:
        return 1
    else:
        return 0

df = pd.read_csv("/users/9/reine097/projects/loes-scoring-2/data/anon_train_scans_and_loes.csv")
output_df = df.copy()
output_df['validation'] = df.apply(is_validation_row, axis=1)
output_df['training'] = df.apply(is_training_row, axis=1)

output_df.to_csv("/users/9/reine097/projects/loes-scoring-2/data/anon_train_scans_and_loes_2.csv")
