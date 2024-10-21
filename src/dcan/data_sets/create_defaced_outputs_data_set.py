from os import listdir
from os.path import isfile, join
import shutil

def is_valid(f):
    return f.endswith('deidentified_SAG_T1_MPRAGE.nii.gz')

my_path = '/users/9/reine097/loes_scoring/s1067-loes-score/Nascene_data/defacing/defaced_outputs/'
only_files = [f for f in listdir(my_path) if isfile(join(my_path, f))]
valid_files = [f for f in only_files if is_valid(f)]

for f in valid_files:
    source_file = join(my_path, f)
    destination_file = join(my_path, 'raw', f)

    shutil.move(source_file, destination_file)

only_files = [f for f in listdir(my_path) if isfile(join(my_path, f))]
for f in only_files:
    source_file = join(my_path, f)
    destination_file = join(my_path, 'not_used', f)

    shutil.move(source_file, destination_file)
