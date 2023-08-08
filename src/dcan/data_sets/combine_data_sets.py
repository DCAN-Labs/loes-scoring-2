import os.path

import pandas as pd

data_set_file_0 = 'data/MNI-space_Loes_data.csv'
data_set_file_1 = "data/Nascene_deID_files.csv"

df_0 = pd.read_csv(data_set_file_0)
df_0["provenance"] = 'igor'
df_0["QC"] = ''
columns_to_keep = ['file', 'QC', 'Gd', 'loes-score', 'provenance']
df_0 = df_0[columns_to_keep]

df_1 = pd.read_csv(data_set_file_1)
df_1 = df_1.rename(columns={"FILE": "file", "loes_score": "loes-score"})
df_1["Gd"] = ''
df_1["provenance"] = 'nascene'
folder = '/home/feczk001/shared/data/loes_scoring/nascene_deid/BIDS/defaced/'
df_1['file'] = df_1['file'].apply(lambda file: os.path.join(folder, file))
df_1 = df_1[columns_to_keep]

combined_df = df_0.append(df_1, ignore_index=True)

combined_df.to_csv('./data/all_data.csv', index=False)
