#!/bin/bash

MRI_IN=$1
MRI_OUT=$2

tempfile=$(mktemp).nii.gz
echo $tempfile

./skull_stripping.sh ${MRI_IN} ${tempfile}
./affine_registration_wrapper.sh ${tempfile} ${MRI_OUT} \
    /home/feczk001/shared/projects/S1067_Loes/data/MNI152/mni_icbm152_t1_tal_nlin_sym_09a_masked.nii.gz
