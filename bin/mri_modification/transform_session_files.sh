#!/bin/bash

SUBJECT=$1
SESSION=$2
STUDY_DIR=/home/feczk001/shared/projects/S1067_Loes/data/niftis_deID/original/
OUT_DIR=/home/feczk001/shared/data/loes_scoring/nascene_deid/BIDS/defaced_atlas_reg

mkdir -p ${OUT_DIR}

for IN in `ls ${STUDY_DIR}/${SUBJECT}/${SESSION}/*.nii.gz`; do
    img_name=`basename $IN`
    echo Registering ${SUBJECT} ${SESSION} ${img_name}
    cmd="./skull_strip_then_register.sh ${IN} ${OUT_DIR}/${SUBJECT}_${SESSION}_space-MNI_${img_name}"
    echo $cmd
    $cmd
done