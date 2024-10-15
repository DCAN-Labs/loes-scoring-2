import pandas as pd
import math
import statistics

file_1 = '/users/9/reine097/loes_scoring/out_1.csv'
file_2 = '/users/9/reine097/loes_scoring/out_2024_10_14.csv'

# Load and sort dataframes by 'prediction'
df_1 = pd.read_csv(file_1).sort_values(by='prediction')
df_2 = pd.read_csv(file_2).sort_values(by='prediction')

# Reset index for both DataFrames to ensure they align numerically
df_1 = df_1.reset_index(drop=True)
df_2 = df_2.reset_index(drop=True)

# Merge both DataFrames on 'file' column, preserving index information
merged_df = pd.merge(df_1[['file']], df_2[['file']], on='file', how='inner', suffixes=('_df1', '_df2'))

# Compute the squared distances between indices
merged_df['index_diff'] = (merged_df.index - merged_df.index_df2) ** 2

# Calculate RMSE and normalized RMSE
rmse = math.sqrt(merged_df['index_diff'].sum() / len(df_1))
normalizer = statistics.mean(range(1, 8))
nrmse = rmse / normalizer

print(nrmse)
