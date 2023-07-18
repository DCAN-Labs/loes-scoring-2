import os.path

import pandas as pd

data_set_file_1 = "data/Nascene_deID_files.csv"

df_1 = pd.read_csv(data_set_file_1)
df_1 = df_1.rename(columns={"FILE": "file", "loes_score": "loes-score"})
df_1["Gd"] = ''
df_1["provenance"] = 'nascene'
print(df_1.QC.unique())
df_1 = df_1.loc[df_1['QC'].isin(['2'])]
dir = '/home/feczk001/shared/data/loes_scoring/nascene_deid/BIDS/defaced/'
df_1['file'] = df_1['file'].apply(lambda file: os.path.join(dir, file))
df_1 = df_1[['file', 'Gd', 'loes-score', 'provenance']]

df_1.to_csv('./data/hospital_data_qc_2.csv', index=False)
