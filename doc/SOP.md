# Standard Operating Procedure

## Preprocessing Workflow Structure

 1. [skull_stripping](../bin/mri_modification/skull_stripping.sh) (singularity)
 2. [affine_registration_wrapper](../bin/mri_modification/affine_registration_wrapper.sh) (flirt)
 3. [normalize_intensity](../bin/mri_modification/normalize_intensity.sh) (fslmaths)

## Creating WM and GM Histograms by Masking
 
 1. [create_masked_files](../src/dcan/eda/create_masked_files.py)
 2. [create_individual_histograms_of_images.py](../src/dcan/eda/create_individual_histograms_of_images.py)
 
## RAVEL
 
See [RAVEL](./RAVEL.Rmd).
 