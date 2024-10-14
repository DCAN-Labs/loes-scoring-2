#!/bin/bash

module load fsl

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <SUBJECT_BRAIN> <MNI_TEMPLATE_BRAIN> <SUBJECT_BRAIN_NORMALIZED>"
    exit 1
fi

SUBJECT_BRAIN=$1
MNI_TEMPLATE_BRAIN=$2
SUBJECT_BRAIN_NORMALIZED=$3

echo "Entering $0"
echo "SUBJECT_BRAIN ${SUBJECT_BRAIN}"
echo "MNI_TEMPLATE_BRAIN ${MNI_TEMPLATE_BRAIN}"
echo "SUBJECT_BRAIN_NORMALIZED ${SUBJECT_BRAIN_NORMALIZED}"
echo

# Echo the values for debugging
echo "Subject mean: $(fslstats ${SUBJECT_BRAIN} -M)"
echo "Subject std: $(fslstats ${SUBJECT_BRAIN} -S)"
echo "Template mean: $(fslstats ${MNI_TEMPLATE_BRAIN} -M)"
echo "Template std: $(fslstats ${MNI_TEMPLATE_BRAIN} -S)"

delimiter="/"
TRANSFORMED_DIR="${SUBJECT_BRAIN_NORMALIZED%${delimiter}*}"
echo "TRANSFORMED_DIR: ${TRANSFORMED_DIR}"
ls ${TRANSFORMED_DIR}
if [ ! -d "$TRANSFORMED_DIR" ]; then
  echo "$TRANSFORMED_DIR does not exist."
  mkdir -p $TRANSFORMED_DIR
fi
ls -ld TRANSFORMED_DIR

cmd="fslmaths ${SUBJECT_BRAIN} \
-sub `fslstats ${SUBJECT_BRAIN} -M` \
-div `fslstats ${SUBJECT_BRAIN} -S` \
-mul `fslstats ${MNI_TEMPLATE_BRAIN} -S` \
-add `fslstats ${MNI_TEMPLATE_BRAIN} -M` \
${SUBJECT_BRAIN_NORMALIZED}"
echo $cmd

# Run the command and check for errors
$cmd
if [ $? -ne 0 ]; then
    echo "Error during normalization"
    exit 1
fi
