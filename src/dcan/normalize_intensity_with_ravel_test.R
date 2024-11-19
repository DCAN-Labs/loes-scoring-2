source("normalize_intensity_with_ravel.R")

input_file_name <- '/home/feczk001/shared/projects/S1067_Loes/data/Fairview-ag/anonymized/processed/sub-01_ses-01_space-MNI_brain_mprage.nii.gz'
output_dir <- '/users/9/reine097/raveled/'
normalize_intensity_with_ravel(input_file_name, output_dir)
