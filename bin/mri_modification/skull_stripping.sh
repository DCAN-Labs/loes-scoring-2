#!/bin/bash

# If you get an error, it may be because you don't have enough memory.

HEAD_WITH_SKULL=${1}
SKULL_STRIPPED_HEAD=${2}

SINGULARITY_FILE=/home/faird/shared/code/external/utilities/synthstrip_1.4.sif
if test -f "$SINGULARITY_FILE" && test -x "$SINGULARITY_FILE";
then
    cd $dir_test
fi

if [ ! -f $SINGULARITY_FILE ]; then
    echo "$SINGULARITY_FILE file not found!"
    exit 1
fi

singularity run -B ${HEAD_WITH_SKULL} \
    $SINGULARITY_FILE -i ${HEAD_WITH_SKULL} -o ${SKULL_STRIPPED_HEAD} --no-csf
