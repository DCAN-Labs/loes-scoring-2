import pandas as pd

df = pd.read_csv('/users/9/reine097/data/fairview-ag/anonymized/skull_stripped_anonymized/loes_scores.csv')
df.head()

sorted_df = df.sort_values('Loes score')
print(sorted)
