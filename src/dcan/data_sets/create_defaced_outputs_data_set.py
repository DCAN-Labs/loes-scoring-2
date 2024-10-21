from os import listdir, makedirs
from os.path import isfile, join
import shutil

def is_valid(filename):
    """Check if the file is valid based on its suffix."""
    return filename.endswith('deidentified_SAG_T1_MPRAGE.nii.gz')

def move_files(files, source_dir, target_subdir):
    """Move files to a target subdirectory within the source directory."""
    target_dir = join(source_dir, target_subdir)
    # Create target directory if it doesn't exist
    makedirs(target_dir, exist_ok=True)

    for filename in files:
        try:
            source_file = join(source_dir, filename)
            destination_file = join(target_dir, filename)
            shutil.move(source_file, destination_file)
            print(f"Moved: {filename} -> {target_subdir}")
        except Exception as e:
            print(f"Error moving {filename}: {e}")

# Define path
my_path = '/users/9/reine097/loes_scoring/s1067-loes-score/Nascene_data/defacing/defaced_outputs/'

# List all files in the directory
all_files = [f for f in listdir(my_path) if isfile(join(my_path, f))]

# Filter valid files
valid_files = [f for f in all_files if is_valid(f)]

# Move valid files to 'raw' subdirectory
move_files(valid_files, my_path, 'raw')

# Move remaining files to 'not_used' subdirectory
remaining_files = [f for f in listdir(my_path) if isfile(join(my_path, f))]
move_files(remaining_files, my_path, 'not_used')
