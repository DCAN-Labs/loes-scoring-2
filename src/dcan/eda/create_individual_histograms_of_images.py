import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
import sys


def get_data(img_dir, img_name):
    """Load and flatten the NIfTI image."""
    # Define the NIfTI file path
    img_path = img_dir / img_name

    # Load NIfTI image
    img = nib.load(img_path)
    gray_matter_data = img.get_fdata()

    # Flatten image data
    flattened_data = gray_matter_data.flatten()

    return flattened_data


def plot_voxel_intensity_histogram(subject_id, img_dir, hist_dir):
    """Generate and save a histogram of voxel intensities for a given subject."""
    try:
        filtered_data_dict = dict()
        for matter_type in ['GM', 'WM']:
            img_name = f'masked-sub-0{subject_id}_ses-01_space-MNI_brain_normalized_{matter_type}_mprage.nii.gz'
            flattened_data = get_data(img_dir, img_name)
            
            # Filter out zero intensities using NumPy
            filtered_data = flattened_data[flattened_data > 0]
            filtered_data_dict[matter_type] = filtered_data

        # Plot histograms
        sns.histplot(filtered_data_dict['WM'], color="blue", label="White matter", kde=True)
        sns.histplot(filtered_data_dict['GM'], color="red", label="Gray matter", kde=True)

        # Set plot title and labels
        plt.title(f'Distribution of voxel intensities for subject {subject_id}')
        plt.xlabel('Voxel intensity')
        plt.ylabel('Frequency')
        plt.legend()

        # Save the plot
        hist_file_name = f'masked-sub-0{subject_id}_ses-01_space-MNI_brain_normalized_voxel_histogram.png'
        hist_path = hist_dir / hist_file_name
        hist_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(hist_path)
        plt.close()  # Close the plot to free memory
        print(f"Histogram saved for subject {subject_id} at {hist_path}")

    except Exception as e:
        print(f"Error processing subject {subject_id}: {e}")


def main(img_dir, good_subjects, hist_dir):
    for subject_id in good_subjects:
        plot_voxel_intensity_histogram(subject_id, img_dir, hist_dir)


if __name__ == "__main__":
    """     How to Run the Script: To run the script, you would now pass the 
            image directory, histogram directory, and the list of subjects as 
            arguments like this:
            
            bash

            python script.py /path/to/img_dir /path/to/hist_dir 1,2,3,4,6,7
    """    
    # Parse command-line arguments
    img_dir_in = Path(sys.argv[1])
    hist_dir_out = Path(sys.argv[2])
    
    # Parse subjects input: comma-separated values as integers
    good_subjects_in = list(map(int, sys.argv[3].split(',')))

    # Call the main function
    main(img_dir_in, good_subjects_in, hist_dir_out)
