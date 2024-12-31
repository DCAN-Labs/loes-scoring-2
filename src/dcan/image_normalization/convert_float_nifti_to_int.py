import nibabel as nib
import numpy as np
import os

dir = '/users/9/reine097/'
# Load the NIfTI file
nifti_file = nib.load(os.path.join(dir, 'mni_icbm152_t1_tal_nlin_sym_09a_mask_float32.nii.gz'))

# Get the image data as a numpy array
data = nifti_file.get_fdata()

# Convert to integers
rounded_data = np.round(data).astype(np.int16)

# Create a new NIfTI image with the integer data
new_nifti = nib.Nifti1Image(rounded_data, nifti_file.affine, nifti_file.header)

# Save the new NIfTI file
nib.save(new_nifti, os.path.join(dir, 'mni_icbm152_t1_tal_nlin_sym_09a_mask_int16.nii.gz'))
