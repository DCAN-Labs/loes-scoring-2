#!/bin/bash

module load fsl

# Example registration command: /panfs/roc/msisoft/fsl/6.0.4/bin/flirt -in /home/feczk001/shared/projects/S1067_Loes/data/Loes_DataLad/sub-1043EATO/ses-20190426/mprage.nii.gz -ref /home/feczk001/shared/projects/S1067_Loes/data/MNI152/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii -out /home/feczk001/shared/projects/S1067_Loes/data/MNI-space_Loes_data/sub-1043EATO_ses-20190426_space_MNI_mprage.nii.gz -dof 9

MRI_IN=$1
MRI_OUT=$2
REF=/home/feczk001/shared/projects/S1067_Loes/data/MNI152/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii

echo Registering ${MRI_IN}
cmd="flirt -in ${MRI_IN} -ref ${REF} -out ${MRI_OUT}"
echo $cmd
$cmd
