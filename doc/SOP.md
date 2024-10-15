# Standard Operating Procedure

## Preprocessing Workflow Structure

 1. [skull_stripping](../bin/mri_modification/skull_stripping.sh) (singularity)
 2. [affine_registration_wrapper](../bin/mri_modification/affine_registration_wrapper.sh) (flirt)
 3. [normalize_intensity](../bin/mri_modification/normalize_intensity.sh) (fslmaths)

 ## Creating WM and GM Histograms by Masking
 TODO
