import pandas as pd

input_file = '/home/miran045/reine097/projects/loes-scoring-2/data/filtered/ashish_all.csv'
df = pd.read_csv(input_file)
print(df.head())
output_file = '/home/miran045/reine097/projects/loes-scoring-2/data/filtered/ashish_all_cleaned_up.csv'
df.to_csv(output_file, index=False)
