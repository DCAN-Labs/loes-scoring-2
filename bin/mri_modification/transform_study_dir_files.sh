#!/bin/bash

CURRENT_DIR=$(pwd)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

STUDY_DIR=$1
OUT_DIR=$2
for sub_directory in $(find ${STUDY_DIR} -mindepth 1 -maxdepth 1 -type d); do
    SUBJECT=$(basename $sub_directory)
    echo "Processing $SUBJECT..."
    for session_directory in $(find ${sub_directory} -mindepth 1 -maxdepth 1 -type d); do
        SESSION=$(basename $session_directory)
        echo "     Processing $SESSION..."
        ./transform_session_files.sh $STUDY_DIR $SUBJECT $SESSION $OUT_DIR
    done
done

cd $CURRENT_DIR
