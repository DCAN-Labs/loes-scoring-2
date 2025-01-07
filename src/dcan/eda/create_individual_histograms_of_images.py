import seaborn as sns
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
import sys
from os import listdir
from os.path import isfile, join

def get_file_identifiers(file_name):
    """Extract subject, session, and run identifiers from the file name."""
    return file_name[:6], file_name[7:13], 'run-01'

def get_data(img_path):
    """Load and flatten the NIfTI image at the specified path, filtering out zero intensities."""
    try:
        img = nib.load(img_path)
        data = img.get_fdata().flatten()
        return data[data > 0]  # Filter out zero intensities directly
    except Exception as e:
        print(f"Error loading NIfTI image at {img_path}: {e}")
        return None

def plot_voxel_intensity_histogram(subject_id, session_id, img_dir, hist_dir, gm_file, wm_file):
    """Generate and save a histogram of voxel intensities for a subject's brain images."""
    hist_data = {}

    gm_img_path = img_dir / gm_file
    wm_img_path = img_dir / wm_file
    hist_data['GM'] = get_data(gm_img_path)
    hist_data['WM'] = get_data(wm_img_path)

    if hist_data:  # Proceed only if data was successfully loaded
        plt.figure()
        for label, data in hist_data.items():
            sns.histplot(data, label=label, kde=True)
        
        plt.title(f'Distribution of voxel intensities for {subject_id}-{session_id}')
        plt.xlabel('Voxel intensity')
        plt.ylabel('Frequency')
        plt.legend()

        # Save histogram
        hist_path = hist_dir / f'masked-sub-{subject_id}_{session_id}_space-MNI_brain_voxel_histogram.png'
        hist_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(hist_path)
        plt.close()
        print(f"Histogram saved for subject {subject_id}-{session_id} at {hist_path}")
    else:
        print(f"No valid data for subject {subject_id}-{session_id}. Skipping histogram generation.")

def main(img_dir, hist_dir):
    img_dir = Path(img_dir)
    hist_dir = Path(hist_dir)

    only_files = [f for f in listdir(img_dir) if isfile(join(img_dir, f))]
    for i in range(0, int(len(only_files) / 2), 2):
        gm_file = only_files[i]
        wm_file = only_files[i + 1]
        subject_id, session_id, _ = get_file_identifiers(gm_file)
        plot_voxel_intensity_histogram(subject_id, session_id, img_dir, hist_dir, gm_file, wm_file)

if __name__ == "__main__":
    """Run the script with the image directory and histogram output directory as arguments."""
    img_dir_in = sys.argv[1]
    hist_dir_out = sys.argv[2]
    main(img_dir_in, hist_dir_out)
