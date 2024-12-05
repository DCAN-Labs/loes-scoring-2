import pandas as pd

def loes_score(row):
   if row['eri_hispanic'] == 1:
      return 'Hispanic'
   if row['eri_afr_amer'] + row['eri_asian'] + row['eri_hawaiian'] + row['eri_nat_amer'] + row['eri_white'] > 1:
      return 'Two Or More'
   if row['eri_nat_amer'] == 1:
      return 'A/I AK Native'
   if row['eri_asian'] == 1:
      return 'Asian'
   if row['eri_afr_amer'] == 1:
      return 'Black/AA'
   if row['eri_hawaiian'] == 1:
      return 'Haw/Pac Isl.'
   if row['eri_white'] == 1:
      return 'White'
   return 'Other'

spreadsheet_file_path = '/users/9/reine097/data/fairview-ag/anonymized/skull_stripped_anonymized/loes_scores.csv'
loes_scores_df = pd.read_csv(spreadsheet_file_path)
plaintext_to_anonymized_file = '/users/9/reine097/data/fairview-ag/anonymized/skull_stripped_anonymized/plaintext_to_anonymized.csv'
plaintext_to_anonymized_df = pd.read_csv(plaintext_to_anonymized_file)
print(plaintext_to_anonymized_df)
plaintext_to_anonymized_df['loes_score'] = plaintext_to_anonymized_df.apply(loes_score, axis=1)
