import pandas as pd

df = pd.read_csv('./data/all/all_data.csv')
qc1_df = df[(df['provenance'] == 'igor') | (df['QC'] == "1")]
qc1_df.to_csv('data/filtered/ashish_and_nascene_qc1.csv', index=False)
