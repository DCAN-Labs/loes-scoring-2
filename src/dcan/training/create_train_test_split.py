import pandas as pd

def every_nth(lst, n):
  """Returns a new list containing every nth element of the input list."""
  return lst[n-1::n]


def create_train_test_split(csv_file_in, csv_file_out):
    csv_data_file = csv_file_in
    df_in = pd.read_csv(csv_data_file)
    df_out = df_in.copy()
    df_out = df_out[~df_out['scan'].str.contains('Gd', na=False)]
    df_out['Gd-enhanced'] = 0
    grouped = df_out.groupby('anonymized_subject_id')['loes-score'].mean()
    df_mean_sorted_asc = grouped.sort_values()
    df_reset = df_mean_sorted_asc.reset_index()
    sorted_subject_list = df_reset['anonymized_subject_id']
    subjects_list = sorted_subject_list.tolist()
    val_subject_list = every_nth(subjects_list, 5)
    trn_subject_list = [subject for subject in sorted_subject_list if subject not in val_subject_list]
    print(f'trn_subject_list: {trn_subject_list}')
    print(f'val_subject_list: {val_subject_list}')

    def process_row_for_training(row):
        return 1 if row['anonymized_subject_id'] in trn_subject_list else 0

    def process_row_for_validation(row):
        return 1 if row['anonymized_subject_id'] in val_subject_list else 0
    
    df_out['training'] = df_out.apply(process_row_for_training, axis=1)
    df_out['validation'] = df_out.apply(process_row_for_validation, axis=1)
    df_out.to_csv(csv_file_out)
