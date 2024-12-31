import nibabel as nib
import numpy as np

# Load the image
img = nib.load('/users/9/reine097/mni_icbm152_t1_tal_nlin_sym_09a_mask.nii.gz')

# Get the image data
data = img.get_fdata()

# Change the data type (e.g., to int16)
new_data = data.astype(np.float32)

# Create a new NIfTI image with the new data type
new_img = nib.Nifti1Image(new_data, img.affine, img.header)

# Save the new image
nib.save(new_img, '/users/9/reine097/mni_icbm152_t1_tal_nlin_sym_09a_mask_float32.nii.gz')
