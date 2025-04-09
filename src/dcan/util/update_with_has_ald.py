import pandas as pd

export_file = '/users/9/reine097/projects/loes-scoring-2/data/anon_train_scans_and_loes_training_test_non_gd_ald.csv'
import_file = '/users/9/reine097/projects/loes-scoring-2/data/anon_train_scans_and_loes_training_test_non_gd.csv'

in_df = pd.read_csv(import_file)
out_df  = in_df.copy()

def has_ald(value):
  if value > 0.0:
    return 1
  else:
    return 0

out_df['has_ald'] = out_df['loes-score'].apply(has_ald)

out_df.to_csv(export_file, index=None)
