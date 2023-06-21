import sys

import pandas as pd
import os.path


def main(data_file):
    for ascending in [True, False]:
        print('Best') if ascending else print('Worst')
        pd.set_option('display.max_colwidth', None)
        df = pd.read_csv(data_file)
        df.sort_values(by=['error'], inplace=True, ascending=ascending)
        df['base_name'] = df.apply(lambda df: os.path.basename(df['file-path']), axis=1)
        best_predictions = df[['base_name', 'prediction', 'error']].head()
        print(best_predictions)

    pd.set_option('display.max_colwidth', None)
    df = pd.read_csv(data_file)
    df['base_name'] = df.apply(lambda df: os.path.basename(df['file-path']), axis=1)
    for ascending in [True, False]:
        if ascending:
            print('### Best predictions by subject')
        else:
            print('### Worst predictions by subject')
        for i in range(1, 8):
            df.sort_values(by=['error'], inplace=True, ascending=ascending)
            subject = f'sub-0{i}'
            print(subject)
            subject_rows = df.loc[df['deidentified_subjectID'] == i]
            print(subject_rows[['base_name', 'prediction', 'error']].head())
            print()

if __name__ == "__main__":
    main(sys.argv[1])
