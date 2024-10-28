import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
import sys
from os import listdir
from os.path import isfile, join


def get_file_identifiers(file_name):
    """Extract subject, session, and run identifiers from the file name."""
    return file_name[:6], file_name[7:13], 'run-01'


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


def plot_voxel_intensity_histogram(subject_id, session_id, img_dir, hist_dir):
    """Generate and save a histogram of voxel intensities for a given subject."""
    try:
        filtered_data_dict = dict()
        for matter_type in ['GM', 'WM']:
            img_name = f'{subject_id}_{session_id}_run-01_space-MNI_{matter_type}_mprage.nii.gz'
            flattened_data = get_data(img_dir, img_name)
            
            # Filter out zero intensities using NumPy
            filtered_data = flattened_data[flattened_data > 0]
            filtered_data_dict[matter_type] = filtered_data

        # Plot histograms
        sns.histplot(filtered_data_dict['WM'], color="blue", label="White matter", kde=True)
        sns.histplot(filtered_data_dict['GM'], color="red", label="Gray matter", kde=True)

        # Set plot title and labels
        plt.title(f'Distribution of voxel intensities for {subject_id}-{session_id}')
        plt.xlabel('Voxel intensity')
        plt.ylabel('Frequency')
        plt.legend()

        # Save the plot
        hist_file_name = f'masked-sub-0{subject_id}_{session_id}_space-MNI_brain_voxel_histogram.png'
        hist_path = hist_dir / hist_file_name
        hist_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(hist_path)
        plt.close()  # Close the plot to free memory
        print(f"Histogram saved for subject {subject_id}_{session_id}_run-01 at {hist_path}")

    except Exception as e:
        print(f"Error processing subject {subject_id}: {e}")


def main(img_dir, hist_dir):
    only_files = [f for f in listdir(img_dir) if isfile(join(img_dir, f))]
    for file_name in only_files:
        subject_id, session_id, _ = get_file_identifiers(file_name)
        plot_voxel_intensity_histogram(subject_id, session_id, img_dir, hist_dir)


if __name__ == "__main__":
    """     How to Run the Script: To run the script, you would now pass the 
            image directory, histogram directory as 
            arguments like this:
            
            bash

            python script.py /path/to/img_dir /path/to/hist_dir
    """    
    # Parse command-line arguments
    img_dir_in = Path(sys.argv[1])
    hist_dir_out = Path(sys.argv[2])

    # Call the main function
    main(img_dir_in, hist_dir_out)
