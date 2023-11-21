# would it be possible to train an AlexNet model to classify CCALD vs. no CCALD?
#
# I would use cases with Loes=0 for no CCALD and loes < 3 for CCALD as a way to filter how many we have for building a
# model
#
# might be worth trying -- just to see if it can work :)
#
# please consult with luci about priorities of course :)

import pandas as pd
import sys

from dcan.training.training import get_subject_from_file_name


def generate_ccald_cutoff(csv_file_path):
    df = pd.read_csv(csv_file_path)
    df['subject'] = df.apply(lambda row: get_subject_from_file_name(row.file), axis=1)
    print(df.head())


if __name__ == '__main__':
    generate_ccald_cutoff(sys.argv[1])
