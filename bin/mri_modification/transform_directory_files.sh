#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 <input_directory> <output_directory>"
    exit 1
}

# Ensure correct number of arguments
if [ "$#" -ne 2 ]; then
    usage
fi

IN_DIR="$1"
OUT_DIR="$2"

# Create the output directory if it doesn't exist
mkdir -p "$OUT_DIR"

# Function to register and transform images
process_image() {
    local input_file="$1"
    local output_file="$2"
    
    local img_name
    img_name=$(basename "$input_file")

    echo "Registering ${img_name}"

    if [ -f "$output_file" ]; then
        echo "File exists, skipping: $output_file"
        return
    else
        echo "File does not exist, processing: $output_file"
    fi

    # Command to transform the image
    local cmd="./perform_transforms.sh \"$input_file\" \"$output_file\""
    echo "$cmd"

    # Execute the transformation command
    eval "$cmd"
}

# Find and process each .nii.gz file in the input directory
find "$IN_DIR" -type f -name "*.nii.gz" | while read -r in_file; do
    output_file="$OUT_DIR/$(basename "$in_file")"
    process_image "$in_file" "$output_file"
done
