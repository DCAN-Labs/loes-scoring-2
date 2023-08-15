import os

import pandas as pd

all_data = pd.read_csv('data/all_data.csv')
ashish_data = all_data.loc[all_data['provenance'] == 'igor']
nascene_data = all_data.loc[all_data['provenance'] == 'nascene']
nascene_data = nascene_data.loc[nascene_data['QC'] == '1']
nascene_data = nascene_data.sample(frac = 1)
n = len(nascene_data)
n1 = n // 2
n2 = n - n1
df1 = nascene_data.iloc[:n1,:]
df2 = nascene_data.iloc[n2:,:]
all_data_1 = pd.concat([ashish_data, df1], ignore_index=True, axis=0)
all_data_2 = pd.concat([ashish_data, df2], ignore_index=True, axis=0)
dir = 'data'
all_data_1.to_csv(os.path.join(dir, 'derived', 'all_data_0.csv'), index=False)
all_data_2.to_csv(os.path.join(dir, 'derived', 'all_data_1.csv'), index=False)
