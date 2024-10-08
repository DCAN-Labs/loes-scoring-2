#!/bin/bash

SUBJECT=$2
SESSION=$3
STUDY_DIR=$1
OUT_DIR=$4

mkdir -p ${OUT_DIR}

for IN in `ls ${STUDY_DIR}/${SUBJECT}/${SESSION}/*.nii.gz`; do
    img_name=`basename $IN`
    echo Registering ${SUBJECT} ${SESSION} ${img_name}
    cmd="./skull_strip_then_register.sh ${IN} ${OUT_DIR}/${SUBJECT}_${SESSION}_space-MNI_brain_${img_name}"
    echo $cmd
    $cmd
done