# Standard Operating Procedure

## Preprocessing Workflow Structure

 1. Preprocessing: [process_study](../../scripts/mri_modification/transform_study_dir_files.sh)

 These scripts form a comprehensive MRI data processing pipeline for neuroimaging analysis. Here's what each script does:

### Overall Workflow
The scripts work together to batch process MRI scans, performing skull stripping and registration to transform brain images into a standardized coordinate space (MNI space).

### Individual Script Functions

**[`transform_study_dir_files.sh`](../../scripts/mri_modification/transform_study_dir_files.sh)** - The main orchestrator script that:
- Processes entire study directories containing multiple subjects
- Iterates through each subject and their sessions
- Calls the session-level processing script for each subject/session combination
- Designed to run on a computing cluster with resource allocation

**[`transform_session_files.sh`](../../scripts/mri_modification/transform_session_files.sh)** - Handles individual subject sessions by:
- Finding all `.nii.gz` files (compressed NIfTI brain images) in a session directory
- Checking if output files already exist to avoid reprocessing
- Calling the transformation pipeline for each brain image
- Creating standardized output filenames with subject, session, and MNI space labels

**[`perform_transforms.sh`](../../scripts/mri_modification/perform_transforms.sh)** - The core processing pipeline that:
- Coordinates the two main processing steps: skull stripping and registration
- Uses temporary files to pass data between processing stages
- Handles error checking and cleanup
- References a standard MNI152 template brain for registration

**[`affine_registration_wrapper.sh`](../../scripts/mri_modification/affine_registration_wrapper.sh)** - Performs spatial normalization using ANTs (Advanced Normalization Tools):
- Registers individual brain images to the MNI152 standard template
- Uses a multi-stage registration approach (Rigid \u2192 Affine \u2192 SyN nonlinear)
- Applies sophisticated image matching metrics and optimization parameters
- Outputs both the registered brain image and transformation matrices

**[`skull_stripping.sh`](../../scripts/mri_modification/skull_stripping.sh)** - Removes non-brain tissue using:
- SynthStrip tool via Singularity container
- Strips skull, scalp, and other non-brain structures
- Prepares clean brain images for accurate registration

### Purpose
This pipeline standardizes MRI brain scans by removing skulls and aligning them to a common coordinate system, which is essential for group-level neuroimaging analyses, allowing researchers to compare brain structure and function across different subjects and studies.

## Creating WM and GM Histograms by Masking
 
 1. This step masks the files with the brain mask.  This removes all non-zero artifacts outside the brain region.
     * [create_brain_masked_files](../../src/dcan/image_normalization/create_brain_masked_files.py), or
     * [create_brain_masked_files_from_csv](../../src/dcan/image_normalization/create_brain_masked_files_from_csv.py)
 3. Optional: [create_individual_histograms_of_images.py](../../src/dcan/eda/create_individual_histograms_of_images.py)
 
## RAVEL
 
See [RAVEL](https://github.com/DCAN-Labs/RAVEL/blob/master/docs/RAVEL.Rmd).

1. The CSF mask file is here 
     */home/feczk001/shared/projects/S1067_Loes/data/MNI152/mni_icbm152_nlin_sym_09a/mni_icbm152_csf_tal_nlin_sym_09a_int_rounded_0_9.nii*
2. You have to create a CSF masked file for each input file.  You can do that with [this code](../../src/dcan/image_normalization/mask_in_csf.py).
3. Create a control region for RAVEL.  [Here](https://github.com/DCAN-Labs/RAVEL/blob/master/R/dcan/create_control_region.R) is example code.
4. Run RAVEL on files.  You can see an example of how to do this [here](https://github.com/DCAN-Labs/RAVEL/blob/master/R/dcan/ravel.R).
