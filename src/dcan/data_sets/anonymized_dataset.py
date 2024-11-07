#!/usr/bin/python

import math
import os
import shutil
import sys
from pathlib import Path
import pandas as pd
from pydeface.utils import deface_image


def get_immediate_subdirectories(directory):
    """Return a list of immediate subdirectories in the given directory."""
    return [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]


def get_ids(folder):
    """Get a dictionary of subjects and their sessions from the folder structure."""
    sessions = {}
    for root, _, _ in os.walk(folder):
        path = root.split(os.sep)
        if len(path) < 3:
            continue
        subject, session = path[1], path[2]
        sessions.setdefault(subject, []).append(session)
    return sessions


def count_sessions(row):
    """Count the number of sessions for a subject."""
    return len(row['Sub_Sessions'])


def format_session_name(row, padding):
    """Format subject name with leading zeros to match the padding length."""
    return f"subject-{str(row['anonymized_subject_name']).zfill(padding)}"


def create_anonymized_session_names(row, session_padding):
    """Create a list of anonymized session names for a subject."""
    return [f"session-{str(i).zfill(session_padding)}" for i in range(len(row['Sub_Sessions']))]


def anonymize_dataset(bids_root_out, csv_path):
    """Anonymize dataset and save mapping to a CSV."""
    non_anonymized_df = pd.read_csv(csv_path)
    anonymized_df = non_anonymized_df.groupby('Sub ID')['Sub_Session'].apply(list).reset_index(name='Sub_Sessions')

    anonymized_df['session-count'] = anonymized_df.apply(count_sessions, axis=1)
    anonymized_df['anonymized_subject_name'] = range(len(anonymized_df))

    subject_padding = int(math.ceil(math.log10(len(anonymized_df))))
    anonymized_df['anonymized_subject_name'] = anonymized_df.apply(format_session_name, args=(subject_padding,), axis=1)

    max_sessions_count = anonymized_df['session-count'].max()
    session_padding = int(math.ceil(math.log10(max_sessions_count)))
    anonymized_df['anonymized_session_dates'] = anonymized_df.apply(
        create_anonymized_session_names, args=(session_padding,), axis=1
    )

    anonymized_df.to_csv(os.path.join(bids_root_out, 'plaintext_to_anonymized.csv'), index=False)
    return anonymized_df


def rename_subject_folders(mapping_df, bids_root_out):
    """Rename subject folders and their sessions to anonymized names."""
    immediate_subdirectories = get_immediate_subdirectories(bids_root_out)
    for subdirectory in immediate_subdirectories:
        result = mapping_df.loc[mapping_df['Sub ID'] == subdirectory[4:]]
        if result.empty:
            continue
        first_row = result.iloc[0]

        sub_sessions, anonymized_sessions = first_row['Sub_Sessions'], first_row['anonymized_session_dates']
        folder = os.path.join(bids_root_out, f"sub-{first_row['Sub ID']}")

        for i, sub_session in enumerate(sub_sessions):
            try:
                old_path = os.path.join(folder, f"ses-{sub_session[-8:]}")
                new_path = os.path.join(folder, anonymized_sessions[i])
                os.rename(old_path, new_path)
            except FileNotFoundError as e:
                print(f"Warning: Failed to rename {old_path} to {new_path}: {e}")

        try:
            os.rename(folder, os.path.join(bids_root_out, first_row['anonymized_subject_name']))
        except FileNotFoundError as e:
            print(f"Warning: Failed to rename subject folder {folder}: {e}")


def deface_images():
    """Run the deface utility for images."""
    deface_image()


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python script.py <source_directory> <target_directory> <csv_path>")
        sys.exit(1)

    source_directory, target_directory, csv_path = sys.argv[1], sys.argv[2], sys.argv[3]
    target_dir_path = Path(target_directory)

    if target_dir_path.exists() and target_dir_path.is_dir():
        shutil.rmtree(target_dir_path)
    shutil.copytree(source_directory, target_directory)

    try:
        encoding = anonymize_dataset(target_directory, csv_path)
        rename_subject_folders(encoding, target_directory)
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        sys.exit(1)
