import seaborn as sns
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
import sys

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

def plot_voxel_intensity_histogram(subject_id, session_id, img_dir, hist_dir):
    """Generate and save a histogram of voxel intensities for a subject's brain images."""
    matter_types = {'WM': "White matter", 'GM': "Gray matter"}
    hist_data = {}

    for matter, label in matter_types.items():
        img_name = f'{subject_id}_{session_id}_run-01_space-MNI_{matter}_mprage_RAVEL.nii.gz'
        img_path = img_dir / img_name

        data = get_data(img_path)
        if data is not None:
            hist_data[label] = data

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

    for img_file in img_dir.glob("*.nii.gz"):
        subject_id, session_id, _ = get_file_identifiers(img_file.name)
        plot_voxel_intensity_histogram(subject_id, session_id, img_dir, hist_dir)

if __name__ == "__main__":
    """Run the script with the image directory and histogram output directory as arguments."""
    img_dir_in = sys.argv[1]
    hist_dir_out = sys.argv[2]
    main(img_dir_in, hist_dir_out)
