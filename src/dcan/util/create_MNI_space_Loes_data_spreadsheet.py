import pandas as pd
from os import listdir
from os.path import isfile, join
import csv

images_dir = '/home/feczk001/shared/projects/S1067_Loes/data/MNI-space_Loes_data'
only_files = [f for f in listdir(images_dir) if isfile(join(images_dir, f))]
with open('./data/MNI-space_Loes_data.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(['file', 'Gd', 'loes-score'])
    df = pd.read_csv('./data/9_7 MRI sessions Igor Loes score updated.csv', skiprows=1)
    for f in only_files:
        parts = f.split('_')
        _, subject = parts[0].split('-')
        _, session = parts[1].split('-')
        subject_session = subject + '_' + session
        row = df.loc[(df['Sub ID'] == subject) & (df['Sub_Session'] == subject_session)]
        if row.empty:
            continue
        loes_score = row.iloc[0]['Unnamed: 35']
        if f[-9:-7] == 'Gd':
            gd = 1
        else:
            gd = 0
        csv_writer.writerow([join(images_dir, f), gd, loes_score])
