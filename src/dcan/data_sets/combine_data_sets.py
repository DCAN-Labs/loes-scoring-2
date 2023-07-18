import os.path

import pandas as pd

data_set_file_0 = 'data/MNI-space_Loes_data.csv'
data_set_file_1 = "data/Nascene_deID_files.csv"

df_0 = pd.read_csv(data_set_file_0)
df_0["GAD (Mil=1, Moderate=2, Severe=3, No contrast=4)"] = ''
df_0["provenance"] = 'igor'
df_0 = df_0[['file', 'Gd', "GAD (Mil=1, Moderate=2, Severe=3, No contrast=4)", 'loes-score', 'provenance']]

df_1 = pd.read_csv(data_set_file_1)
df_1 = df_1.rename(columns={"FILE": "file", "loes_score": "loes-score"})
df_1["Gd"] = ''
df_1["provenance"] = 'nascene'
print(df_1.QC.unique())
df_1 = df_1.loc[df_1['QC'] == '1']
dir = '/home/feczk001/shared/data/loes_scoring/nascene_deid/BIDS/defaced/'
df_1['file'] = df_1['file'].apply(lambda file: os.path.join(dir, file))
df_1 = df_1[['file', 'Gd', 'loes-score', 'provenance']]

combined_df = df_0.append(df_1, ignore_index=True)

combined_df.to_csv('./data/all_loes_score_data.csv', index=False)
