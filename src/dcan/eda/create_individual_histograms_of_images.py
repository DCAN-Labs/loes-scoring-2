import os
import nibabel
import seaborn as sns
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
import sys
import nilearn
from nilearn.image import load_img
import matplotlib
import tqdm
matplotlib.use('TkAgg',force=True)
from matplotlib import pyplot as plt
print("Switched to:",matplotlib.get_backend())


EPSILON = 0.01


def get_file_identifiers(file):
    """
    Extract subject, session, and run identifiers from the file name.
    Assumes a specific naming convention.
    """
    try:
        file_name = file.name
        subject_id = file_name[:10]
        session_id = file_name[11:21]
        run_id = "run-01"  # Default run identifier
        return subject_id, session_id, run_id
    except IndexError as e:
        raise ValueError(f"Invalid file name format: {file_name}. Error: {e}")


def get_data(img_path):
    """
    Load and flatten the NIfTI image at the specified path, filtering out zero intensities.
    """
    try:
        img = nib.load(img_path)
        data = img.get_fdata().flatten()
        return data[data > 0]  # Filter out zero intensities
    except Exception as e:
        print(f"Error loading NIfTI image at {img_path}: {e}")
        return None


def plot_voxel_intensity_histogram(subject_id, session_id, img_dir, hist_dir, gm_file, wm_file):
    """
    Generate and save a histogram of voxel intensities for a subject's brain images.
    """
    gm_brain_data_masked_img = mask_in_matter_data(img_dir, subject_id, session_id, gm_file)
    wm_brain_data_masked_img = mask_in_matter_data(img_dir, subject_id, session_id, wm_file)

    if gm_brain_data_masked_img is None or wm_brain_data_masked_img is None:
        print(f"Data missing for subject {subject_id}-{session_id}. Skipping histogram generation.")
        return

    # Plot histograms
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.set_xlim([0, 600])
    sns.histplot(gm_brain_data_masked_img.flatten(), label="GM", kde=True, color="blue", alpha=0.6)
    sns.histplot(wm_brain_data_masked_img.flatten(), label="WM", kde=True, color="orange", alpha=0.6)
    plt.title(f"Distribution of Voxel Intensities for {subject_id}-{session_id}")
    plt.xlabel("Voxel Intensity")
    plt.ylabel("Frequency")
    plt.legend()

    # Save histogram plot
    hist_path = hist_dir / f"masked-{subject_id}_{session_id}_space-MNI_brain_voxel_histogram.png"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(hist_path, dpi=300, bbox_inches="tight")
    plt.close()

def mask_in_matter_data(img_dir, subject_id, session_id, m_file):
    path_to_brain = os.path.join(img_dir, f'{subject_id}_{session_id}_run-00_space-MNI_mprage_RAVEL.nii.gz')
    brain = nilearn.image.load_img(path_to_brain)
    brain_data = brain.get_fdata()

    m_mask = nilearn.image.load_img(m_file)
    m_mask_data = m_mask.get_fdata()
    m_mask_indexes=(m_mask_data==1)

    m_brain_data_masked=brain_data * m_mask_indexes
    # Filter out zero values
    filtered_data = m_brain_data_masked[m_brain_data_masked > EPSILON]
    
    return filtered_data


def main(img_dir, hist_dir, wm_mask, gm_mask):
    """
    Main function to process all files and generate histograms for each subject.
    """
    img_dir = Path(img_dir)
    hist_dir = Path(hist_dir)

    # List all NIfTI files in the image directory
    files = sorted(f for f in img_dir.glob("*_RAVEL.nii.gz"))

    for file in tqdm.tqdm(files):
        try:
            subject_id, session_id, _ = get_file_identifiers(file)
            plot_voxel_intensity_histogram(subject_id, session_id, img_dir, hist_dir, wm_mask, gm_mask)
        except ValueError as e:
            print(f"Error processing file: {e}")
            continue


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python script.py <image_directory> <histogram_output_directory> <wm_mask> <gm_mask>")
        sys.exit(1)

    img_dir_in = sys.argv[1]
    hist_dir_out = sys.argv[2]
    wm_mask = sys.argv[3]
    gm_mask = sys.argv[4]

    try:
        main(img_dir_in, hist_dir_out, wm_mask, gm_mask)
    except Exception as e:
        print(f"Critical error: {e}")
        sys.exit(1)
