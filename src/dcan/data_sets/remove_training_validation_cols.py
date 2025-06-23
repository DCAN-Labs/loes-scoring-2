import pandas as pd

df = pd.read_csv('/users/9/reine097/projects/loes-scoring-2/data/regression_training_validation.csv')
df_modified = df.drop(columns=['training', 'validation', 'Unnamed: 0'])
df_modified.to_csv('/users/9/reine097/projects/loes-scoring-2/data/regression_training_validation.csv', index=False)
