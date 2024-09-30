#!/bin/bash

LOES_SCORING=/users/9/reine097/loes_scoring
HEAD=${LOES_SCORING}/Loes_score/sub-1043EATO/ses-20190426/mprage.nii.gz
DESKULLED_DIR=${LOES_SCORING}/deskulled
SKULL_STRIPPED=${DESKULLED_DIR}/Loes_score/sub-1043EATO/ses-20190426/mprage.nii.gz

cd $DESKULLED_DIR

mkdir -p Loes_score/sub-1043EATO/ses-20190426

singularity run -B ${HEAD} /home/faird/shared/code/external/utilities/synthstrip_1.4.sif -i ${HEAD} -o ${SKULL_STRIPPED} --no-csf
