# Standard Operating Procedure

## Preprocessing Workflow Structure

 1. Preprocessing: [process_study](../bin/mri_modification/transform_study_dir_files.sh)

## Creating WM and GM Histograms by Masking
 
 1. This step masks the files with the brain mask.  This removes all non-zero artifacts outside the brain region.
     * [create_brain_masked_files](../../src/dcan/image_normalization/create_brain_masked_files.py), or
     * [create_brain_masked_files_from_csv](../../src/dcan/image_normalization/create_brain_masked_files_from_csv.py)
 3. Optional: [create_individual_histograms_of_images.py](../../src/dcan/eda/create_individual_histograms_of_images.py)
 
## RAVEL
 
See [RAVEL](https://github.com/DCAN-Labs/RAVEL/blob/master/docs/RAVEL.Rmd).

1. The CSF mask file is here 
     */home/feczk001/shared/projects/S1067_Loes/data/MNI152/mni_icbm152_nlin_sym_09a/*
2. You have to create a CSF masked file for each input file.  You can do that with [this code](../../src/dcan/image_normalization/mask_in_csf.py).
3. Create a control region for RAVEL.  [Here](https://github.com/DCAN-Labs/RAVEL/blob/master/R/dcan/create_control_region.R) is example code.
4. Run RAVEL on files.  You can see an example of how to do this [here](https://github.com/DCAN-Labs/RAVEL/blob/master/R/dcan/ravel.R).
