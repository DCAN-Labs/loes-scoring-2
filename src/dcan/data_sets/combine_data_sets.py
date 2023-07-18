import pandas as pd

data_set_file_0 = 'data/MNI-space_Loes_data.csv'
data_set_file_1 = "/home/feczk001/shared/data/loes_scoring/nascene_deid/BIDS/loes_scores_gd_model.csv"

df_0 = pd.read_csv(data_set_file_0)
df_0["GAD (Mil=1, Moderate=2, Severe=3, No contrast=4)"] = ''
df_0 = df_0[['file', 'Gd', "GAD (Mil=1, Moderate=2, Severe=3, No contrast=4)", 'loes-score']]

df_1 = pd.read_csv(data_set_file_1)
df_1 = df_1.rename(columns={"file-path": "file", "loes_score": "loes-score"})
df_1["Gd"] = ''
df_1 = df_1[['file', 'Gd', "GAD (Mil=1, Moderate=2, Severe=3, No contrast=4)", 'loes-score']]

combined_df = df_0.append(df_1, ignore_index=True)

combined_df.to_csv('./all_loes_score_data.csv')
