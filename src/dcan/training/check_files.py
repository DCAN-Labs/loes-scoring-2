import os
import os.path
from fastMONAI.vision_all import *

loes_scoring_folder = '/home/feczk001/shared/data/loes_scoring/'

df = pd.read_csv(os.path.join(loes_scoring_folder, 'Nascene_deID_files.csv'))

defaced_folder = os.path.join(loes_scoring_folder, 'nascene_deid/BIDS/defaced/')

def add_folder(row):
    return os.path.join(defaced_folder, row['FILE'])


def file_exists(row):
    fname = row['full_path']

    if os.path.isfile(fname):
        return 1
    else:
        return 0


df['full_path'] = df.apply(add_folder, axis=1)
df['file_exists'] = df.apply(file_exists, axis=1)
df = df[df.file_exists == 1]

df.to_csv(os.path.join(loes_scoring_folder, 'Nascene_deID_existing_files.csv'))
