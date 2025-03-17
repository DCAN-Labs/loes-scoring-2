#!/bin/bash

# Batch processing script for MRI data
# Run with srun for resource allocation. Example:
# srun --time=8:00:00 --mem=32GB -c 8 --tmp=20gb -p interactive -A feczk001 --x11 --pty bash

# usage example:
#   ./process_study.sh /path/to/study_dir /path/to/output_dir

# Ensure proper usage
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <STUDY_DIR> <OUT_DIR>"
    exit 1
fi

# Parse input arguments
STUDY_DIR=$1
OUT_DIR=$2

# Check if input directories exist
if [ ! -d "$STUDY_DIR" ]; then
    echo "Error: STUDY_DIR does not exist: $STUDY_DIR"
    exit 1
fi

if [ ! -d "$OUT_DIR" ]; then
    echo "Creating output directory: $OUT_DIR"
    mkdir -p "$OUT_DIR" || {
        echo "Error: Could not create OUT_DIR: $OUT_DIR"
        exit 1
    }
fi

# Store current and script directory
CURRENT_DIR=$(pwd)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR" || {
    echo "Error: Could not change to script directory: $SCRIPT_DIR"
    exit 1
}

# Process each subject directory
for sub_directory in "$STUDY_DIR"/*/; do
    if [ -d "$sub_directory" ]; then
        SUBJECT=$(basename "$sub_directory")
        echo "Processing $SUBJECT..."

        # Process each session directory for the subject
        for session_directory in "$sub_directory"/*/; do
            if [ -d "$session_directory" ]; then
                SESSION=$(basename "$session_directory")
                echo "    Processing $SESSION..."
                ./transform_session_files.sh "$STUDY_DIR" "$SUBJECT" "$SESSION" "$OUT_DIR" || {
                    echo "Error processing $SUBJECT - $SESSION"
                    continue
                }
            fi
        done
    fi
done

# Return to the original directory
cd "$CURRENT_DIR" || {
    echo "Error: Could not return to original directory: $CURRENT_DIR"
    exit 1
}

echo "Batch processing completed successfully."
