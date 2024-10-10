#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <MRI_IN> <MRI_OUT>"
    exit 1
fi

MRI_IN="$1"
MRI_OUT="$2"

tempfile=$(mktemp --suffix=.nii.gz)
echo "tempfile: ${tempfile}"
ls -l "${tempfile}"
tempfile_2=$(mktemp --suffix=.nii.gz)

REF="/home/feczk001/shared/projects/S1067_Loes/data/MNI152/mni_icbm152_t1_tal_nlin_sym_09a_masked.nii.gz"

# Run the processing scripts and check for errors
if ! ./skull_stripping.sh "${MRI_IN}" "${tempfile}"; then
    echo "Error in skull_stripping.sh"
    exit 1
fi

if ! ./affine_registration_wrapper.sh "${tempfile}" "${tempfile_2}" "${REF}"; then
    echo "Error in affine_registration_wrapper.sh"
    exit 1
fi

if ! ./normalize_intensity.sh "${tempfile_2}" "${REF}" "${MRI_OUT}"; then
    echo "Error in normalize_intensity.sh"
    exit 1
fi

echo "Processing completed successfully."
