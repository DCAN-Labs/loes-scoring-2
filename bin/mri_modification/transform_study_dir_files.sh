#!/bin/bash

# STUDY_DIR=/home/feczk001/shared/projects/S1067_Loes/data/niftis_deID/original/
# OUT_DIR=/home/feczk001/shared/data/loes_scoring/nascene_deid/BIDS/defaced_atlas_reg

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
