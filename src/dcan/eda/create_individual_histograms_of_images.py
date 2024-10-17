import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path

def get_data(img_dir, img_name):
    # Define the NIfTI file path
    img_path = img_dir / img_name

    # Load NIfTI image
    img = nib.load(img_path)
    gray_matter_data = img.get_fdata()

    # Flatten image data and filter out zero intensities
    flattened_data = gray_matter_data.flatten()

    return flattened_data


def plot_voxel_intensity_histogram(subject_id, img_dir, hist_dir):
    """Generate and save a histogram of voxel intensities for a given subject."""
    try:
        filtered_data_dict = dict()
        for matter_type in ['GM', 'WM']:
            img_name = f'masked-sub-0{subject_id}_ses-01_space-MNI_brain_normalized_{matter_type}_mprage.nii.gz'
            flattened_data = get_data(img_dir. img_name)
            filtered_data = [flattened_data[i] for i in range(len(flattened_data)) if flattened_data[i] > 0]
            filtered_data_dict[matter_type] = filtered_data
        df = pd.DataFrame({'white_matter_voxel_intensity': filtered_data_dict['WM'], 'gray_matter_voxel_intensity': filtered_data_dict['GM']})
        # Plot histogram
        fig, ax = plt.subplots()
        sns.histplot(data=df['WM'], ax=ax, color="blue", label="White matter", kde=True)
        sns.histplot(data=df['GM'], ax=ax, color="blue", label="Gray matter", kde=True)
        plt.title(f'Distribution of gray matter voxel intensities in {img_name}')
        plt.xlabel('Voxel intensity')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)

        # Save the plot
        hist_file_name = f'{img_name[:-7]}.png'  # Strip '.nii.gz' from the name
        hist_path = hist_dir / hist_file_name
        hist_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(hist_path)
        plt.close()  # Close the plot to free memory
        print(f"Histogram saved for subject {subject_id} at {hist_path}")

    except Exception as e:
        print(f"Error processing subject {subject_id}: {e}")

def main():
    good_subjects = [1, 2, 4]  # List of good subjects
    img_dir = Path('/home/feczk001/shared/projects/S1067_Loes/data/niftis_deID/masked/non_gd/GM/')
    hist_dir = Path('/home/feczk001/shared/projects/S1067_Loes/data/niftis_deID/histograms/non_gd/gm/')

    for subject_id in good_subjects:
        plot_voxel_intensity_histogram(subject_id, img_dir, hist_dir)

if __name__ == "__main__":
    main()
