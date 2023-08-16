import pandas as pd

df = pd.read_csv('./data/all/all_data.csv')
df = df[df['Gd'].notna()]
df = df.astype({'Gd': 'int32'})
qc1_df = df[(df['provenance'] == 'igor') & (df['Gd'] == 1)]
qc1_df.to_csv('data/filtered/ashish_gd.csv', index=False)
