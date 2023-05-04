#!/bin/bash


module load fsl


# Example registration command: /panfs/roc/msisoft/fsl/6.0.4/bin/flirt -in /home/feczk001/shared/projects/S1067_Loes/data/Loes_DataLad/sub-1043EATO/ses-20190426/mprage.nii.gz -ref /home/feczk001/shared/projects/S1067_Loes/data/MNI152/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii -out /home/feczk001/shared/projects/S1067_Loes/data/MNI-space_Loes_data/sub-1043EATO_ses-20190426_space_MNI_mprage.nii.gz -dof 9


SUBJECT=$1
SESSION=$2
STUDY_DIR=/home/feczk001/shared/projects/S1067_Loes/data/Loes_DataLad
REF=/home/feczk001/shared/projects/S1067_Loes/data/MNI152/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii
OUT_DIR=/home/feczk001/shared/projects/S1067_Loes/data/MNI-space_Loes_data

for IN in `ls ${STUDY_DIR}/${SUBJECT}/${SESSION}/*.nii.gz`; do
    img_name=`basename $IN`
    echo Registering ${SUBJECT} ${SESSION} ${img_name}
    cmd="flirt -in ${IN} -ref ${REF} -out ${OUT_DIR}/${SUBJECT}_${SESSION}_space-MNI_${img_name}"
    echo $cmd
    $cmd
done




