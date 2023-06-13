import os
import shutil

import pandas as pd

atlas_reg_dir = '/home/feczk001/shared/data/loes_scoring/nascene_deid/BIDS/atlas_reg'

# Sample command
# ./affine_registration_wrapper.sh sub-02 20090924

rootdir = atlas_reg_dir
files = (file for file in os.listdir(rootdir)
         if os.path.isfile(os.path.join(rootdir, file)))
old_sub = None
session_index = 0
df = pd.DataFrame(
    {
        "subject": [],
        "session-date": [],
        "session_number": [],
        "file-path": []
    }
)
i = 0
anonymized_names_dir = '/home/feczk001/shared/data/loes_scoring/nascene_deid/BIDS/anonymized_names/'
old_session_date = None
for nifti_file in files:
    this_sub = nifti_file[4:6]
    session_date = nifti_file[7:15]
    if this_sub != old_sub:
        # new subject
        old_sub = this_sub
        session_index = 0
        old_session_date = session_date
    else:
        if session_date != old_session_date:
            old_session_date = session_date
            session_index += 1
    anonymized_file_name = nifti_file.replace(session_date, f"session-{session_index:02d}", 1)
    anonymized_file_name = anonymized_file_name.replace(session_date, f"", 1)
    anonymized_file_name = f'{anonymized_file_name[:-14]}.nii.gz'
    print(f'old: {nifti_file}')
    src = os.path.join(atlas_reg_dir, nifti_file)
    dst = os.path.join(anonymized_names_dir, anonymized_file_name)
    shutil.copyfile(src, dst)
    print(f'new: {anonymized_file_name}\n')
    df.loc[i] = [old_sub, session_date, str(int(session_index)), dst]
    i += 1
df.to_csv(os.path.join(anonymized_names_dir, 'loes_scores.csv'), index=False)
