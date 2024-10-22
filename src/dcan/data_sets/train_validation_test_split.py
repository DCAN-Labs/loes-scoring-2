import os
import pandas as pd
import os.path

csv_file = '/users/9/reine097/loes_scoring/s1067-loes-score/Nascene_data/defacing/defaced_outputs/loes_scores.csv'

def file_exists(file):
    dir = '/users/9/reine097/loes_scoring/s1067-loes-score/Nascene_data/defacing/defaced_outputs/preprocessed/'
    fname = os.path.join(dir, file)
    
    return os.path.isfile(fname)

df = pd.read_csv(csv_file)

# Apply the boolean function to the column
df = df[df['file'].apply(file_exists)]

print(df)
