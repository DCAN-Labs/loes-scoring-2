import os
import pandas as pd

# Define paths
csv_file = '/users/9/reine097/loes_scoring/s1067-loes-score/Nascene_data/defacing/defaced_outputs/loes_scores.csv'
base_dir = '/users/9/reine097/loes_scoring/s1067-loes-score/Nascene_data/defacing/defaced_outputs/preprocessed/'

# Check if the file exists in the preprocessed directory
def file_exists(file_name):
    return os.path.isfile(os.path.join(base_dir, file_name))

# Load the CSV into a DataFrame
df = pd.read_csv(csv_file)

# Filter the DataFrame by checking if the files exist in the specified directory
df['file_exists'] = df['file'].apply(file_exists)

# Only keep rows where the file exists
df_filtered = df[df['file_exists']]

grouped = df.groupby(['subject', 'subject'])

result = grouped.agg({'loes_score': 'mean'}) 

print(result)
