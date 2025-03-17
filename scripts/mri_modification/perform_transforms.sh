#!/bin/bash

# Script for MRI Processing Workflow
# Author: Paul Reiners
# Date: 12/10/2024

# Ensure the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <MRI_IN> <MRI_OUT>"
    exit 1
fi

# Parse command-line arguments
MRI_IN="$1"
MRI_OUT="$2"

# Constants
REF="/home/feczk001/shared/projects/S1067_Loes/data/MNI152/mni_icbm152_t1_tal_nlin_sym_09a_masked.nii.gz"
REG="registered_"

# Temporary file for intermediate processing
tempfile=$(mktemp --suffix=.nii.gz)

# Store and switch directories
original_dir=$(pwd)
script_dir=$(dirname "$0")

# Navigate to the script's directory
cd "$script_dir" || {
    echo "Error: Could not change to script directory: $script_dir"
    exit 1
}

echo "Working in script directory: $script_dir"

# Run skull stripping and check for errors
if ! ./skull_stripping.sh "$MRI_IN" "$tempfile"; then
    echo "Error: skull_stripping.sh failed"
    rm -f "$tempfile"  # Clean up temporary file
    exit 1
fi

# Run affine registration and check for errors
if ! ./affine_registration_wrapper.sh "$tempfile" "$MRI_OUT" "$REF" "$REG"; then
    echo "Error: affine_registration_wrapper.sh failed"
    rm -f "$tempfile"  # Clean up temporary file
    exit 1
fi

# Remove temporary file
rm -f "$tempfile"

# Return to the original directory
cd "$original_dir" || {
    echo "Error: Could not return to original directory: $original_dir"
    exit 1
}

echo "Back in original directory: $original_dir"
echo "Processing completed successfully."
