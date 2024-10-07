#!/bin/bash

# If you get an error, it may be because you don't have enough memory.

HEAD_WITH_SKULL=${1}
SKULL_STRIPPED_HEAD=${2}

singularity run -B ${HEAD_WITH_SKULL} \
    /home/faird/shared/code/external/utilities/synthstrip_1.4.sif \
    -i ${HEAD_WITH_SKULL} -o ${SKULL_STRIPPED_HEAD} --no-csf
