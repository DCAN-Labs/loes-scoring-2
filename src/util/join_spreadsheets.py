import os

import pandas as pd
from datetime import datetime


def get_session_id(date_of_mri):
    dt = datetime.strptime(date_of_mri, '%m/%d/%Y')
    date_time = dt.strftime("%Y%m%d")  # <- here
    return date_time


nascene_deid_dir = '/home/feczk001/shared/data/loes_scoring/nascene_deid'
bids_dir = os.path.join(nascene_deid_dir, 'BIDS')
loes_score_file = os.path.join(bids_dir, 'loes_scores.csv')
loes_score_df = pd.read_csv(loes_score_file)
data_mappings_file = os.path.join(nascene_deid_dir, 'DICOMS/data_mappings.tsv')
subject_id_mappings_df = pd.read_csv(data_mappings_file, sep='\t')


def remove_cruft_from_subject_id(subject_id):
    if subject_id.endswith('.zip'):
        subject_id = subject_id[:-4]
    if subject_id.endswith('_part1') or subject_id.endswith('_part2'):
        subjectID = subject_id[:-6]

    return subject_id


def remove_cruft_from_deidentified_subject_id(subjectID):
    subjectID = subjectID[5:]
    if subjectID.endswith('a') or subjectID.endswith('b'):
        subjectID = subjectID[:-1]

    return int(subjectID)


subject_id_mappings_df['subjectIDTemp'] = \
    subject_id_mappings_df.apply(lambda row: remove_cruft_from_subject_id(row['subjectID']), axis=1)
subject_id_mappings_df = subject_id_mappings_df.drop('subjectID', axis=1)
subject_id_mappings_df['deidentified_subjectID'] = \
    subject_id_mappings_df.apply(lambda row: remove_cruft_from_deidentified_subject_id(row['deidentified_subjectID']),
                                 axis=1)
subject_id_mappings_df.drop_duplicates(inplace=True)
subject_id_mappings_df.rename(columns={'subjectIDTemp': 'subjectID'}, inplace=True)
loes_score_df.rename(columns={'subject': 'deidentified_subjectID'}, inplace=True)
loes_score_df = pd.merge(subject_id_mappings_df, loes_score_df, on='deidentified_subjectID', how='inner')
master_file_11_28_22_ald_loes_scores_dn_file = os.path.join(bids_dir, 'Master file_11_28_22 ALD Loes scores DN.csv')
master_file_11_28_22_ald_loes_scores_dn_df = pd.read_csv(master_file_11_28_22_ald_loes_scores_dn_file)
master_file_11_28_22_ald_loes_scores_dn_df.ffill(axis=0, inplace=True)
master_file_11_28_22_ald_loes_scores_dn_df['session'] = \
    master_file_11_28_22_ald_loes_scores_dn_df.apply(lambda row: get_session_id(row['Date of MRI']), axis=1)
master_file_11_28_22_ald_loes_scores_dn_df_columns = master_file_11_28_22_ald_loes_scores_dn_df.columns
named_columns = [column for column in master_file_11_28_22_ald_loes_scores_dn_df_columns if
                 not column.startswith('Unnamed: ')]
master_file_11_28_22_ald_loes_scores_dn_df = master_file_11_28_22_ald_loes_scores_dn_df[[*named_columns]]
loes_score_df.rename(columns={'session-date': 'session', 'subjectID': 'subject'}, inplace=True)
loes_score_df = loes_score_df.astype({'session': 'string'})
master_file_11_28_22_ald_loes_scores_dn_df.rename(columns={'Subject ID': 'subject', 'Loes Score:': 'loes_score'},
                                                  inplace=True)
loes_score_df = pd.merge(master_file_11_28_22_ald_loes_scores_dn_df, loes_score_df, on=['subject', 'session'])
loes_score_df.to_csv(os.path.join(bids_dir, 'loes_scores_all.csv'), index=False)
