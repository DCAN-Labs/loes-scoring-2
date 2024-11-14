#!/bin/bash

SUBJECT=$2
SESSION=$3
STUDY_DIR=$1
OUT_DIR=$4

mkdir -p ${OUT_DIR}

for IN in `ls ${STUDY_DIR}/${SUBJECT}/${SESSION}/*.nii.gz`; do
    img_name=`basename $IN`
    echo Registering ${SUBJECT} ${SESSION} ${img_name}
    file_path=${OUT_DIR}${SUBJECT}_${SESSION}_space-MNI_brain_${img_name}
    if [ -f "$file_path" ]; then
        echo "File exists, skipping..."
    else
        echo "File does not exist, proceeding..."
        cmd="./perform_transforms.sh ${IN} ${file_path}"
        echo $cmd
        $cmd
    fi
done