import sys
import nibabel as nib
import numpy as np
import os

def main(dir, input_nifti_file, output_nifti_file):
    # Load the NIfTI file
    nifti_file = nib.load(os.path.join(dir, input_nifti_file))

    # Get the image data as a numpy array
    data = nifti_file.get_fdata()
    data_copy = data.copy()

    # Convert to integers
    rounded_data = np.round(data_copy).astype(np.int16)

    # Create a new NIfTI image with the integer data
    new_nifti = nib.Nifti1Image(rounded_data, nifti_file.affine, nifti_file.header)

    # Save the new NIfTI file
    nib.save(new_nifti, os.path.join(dir, output_nifti_file))

if __name__ == "__main__":
    dir = sys.argv[1]
    input_nifti_file = sys.argv[2]
    output_nifti_file = sys.argv[3]
    main(dir, input_nifti_file, output_nifti_file)
