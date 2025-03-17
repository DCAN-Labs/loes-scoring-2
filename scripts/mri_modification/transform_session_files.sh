#!/bin/bash

# Validate arguments
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <study_dir> <subject> <session> <out_dir>"
    exit 1
fi

STUDY_DIR="$1"
SUBJECT="$2"
SESSION="$3"
OUT_DIR="$4"

# Create output directory if it doesn't exist
mkdir -p "$OUT_DIR"

# Store and switch directories
original_dir=$(pwd)
script_dir=$(dirname "$0")

# Navigate to the script's directory
if ! cd "$script_dir"; then
    echo "Error: Could not change to script directory: $script_dir"
    exit 1
fi

echo "Working in script directory: $script_dir"

# Find and process .nii.gz files
find "${STUDY_DIR}/${SUBJECT}/${SESSION}" -type f -name "*.nii.gz" | while IFS= read -r IN; do
    img_name=$(basename "$IN")
    echo "Registering ${SUBJECT} ${SESSION} ${img_name}"
    
    file_path="${OUT_DIR}/${SUBJECT}_${SESSION}_space-MNI_brain_${img_name}"
    if [ -f "$file_path" ]; then
        echo "File exists, skipping: $file_path"
    else
        echo "File does not exist, proceeding..."
        cmd="./perform_transforms.sh \"$IN\" \"$file_path\""
        echo "$cmd"
        eval "$cmd"
    fi
done

# Return to the original directory
if ! cd "$original_dir"; then
    echo "Error: Could not return to original directory: $original_dir"
    exit 1
fi

echo "Back in original directory: $original_dir"
