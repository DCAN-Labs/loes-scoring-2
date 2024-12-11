#!/bin/bash

SUBJECT=$2
SESSION=$3
STUDY_DIR=$1
OUT_DIR=$4

mkdir -p ${OUT_DIR}

# Store and switch directories
original_dir=$(pwd)
script_dir=$(dirname "$0")

# Navigate to the script's directory
cd "$script_dir" || {
    echo "Error: Could not change to script directory: $script_dir"
    exit 1
}

echo "Working in script directory: $script_dir"

for IN in `ls ${STUDY_DIR}/${SUBJECT}/${SESSION}/*.nii.gz`; do
    img_name=`basename $IN`
    echo Registering ${SUBJECT} ${SESSION} ${img_name}
    file_path=${OUT_DIR}/${SUBJECT}_${SESSION}_space-MNI_brain_${img_name}
    if [ -f "$file_path" ]; then
        echo "File exists, skipping..."
    else
        echo "File does not exist, proceeding..."
        cmd="./perform_transforms.sh ${IN} ${file_path}"
        echo $cmd
        $cmd
    fi
done

# Return to the original directory
cd "$original_dir" || {
    echo "Error: Could not return to original directory: $original_dir"
    exit 1
}

echo "Back in original directory: $original_dir"
