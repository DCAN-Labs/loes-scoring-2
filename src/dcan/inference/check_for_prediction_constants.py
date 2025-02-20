import pandas as pd

file1 = '/users/9/reine097/projects/loes-scoring-2/doc/models/model12/256_epochs/model12_256.csv'
file2 = '/users/9/reine097/projects/loes-scoring-2/doc/models/model12/512_epochs/model12_512.csv'

for file in [file1, file2]:
    df1 = pd.read_csv(file) 
    filtered_df1 = df1[(df1['loes-score'] < 0.01) & df1['predicted_loes_score'].notna()]
    num_rows = len(filtered_df1)
    print(num_rows)
    unique_values1 = filtered_df1['predicted_loes_score'].unique()
    print(f'unique_values: {unique_values1}')
