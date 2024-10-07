#!/bin/bash

# If you get an error, it may be because you don't have enough memory.

DIR=/home/feczk001/shared/projects/S1067_Loes/data/niftis_deID/atlas_reg
HEAD=${DIR}/sub-01_session-01_space-MNI_mprage.nii.gz
SKULL_STRIPPED=~/sub-01_session-01_space-MNI_mprage-deskulled.nii.gz

cd $DIR

singularity run -B ${HEAD} /home/faird/shared/code/external/utilities/synthstrip_1.4.sif -i ${HEAD} -o ${SKULL_STRIPPED} --no-csf
