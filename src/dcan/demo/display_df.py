import pandas as pd

data_file = 'data/anon_train_scans_and_loes_training_test_non_gd.csv'
df = pd.read_csv(data_file)
df_new = df.drop(columns=['training', 'validation'], axis=1) 
print(df_new.head())
print(df_new.info())
df_new.to_csv(data_file, index=False)
