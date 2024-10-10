#!/bin/bash

module load fsl

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <SUBJECT_BRAIN> <MNI_TEMPLATE_BRAIN> <SUBJECT_BRAIN_NORMALIZED>"
    exit 1
fi

SUBJECT_BRAIN=$1
MNI_TEMPLATE_BRAIN=$2
SUBJECT_BRAIN_NORMALIZED=$3

echo "Normalizing ${SUBJECT_BRAIN}"

# Echo the values for debugging
echo "Subject mean: $(fslstats ${SUBJECT_BRAIN} -M)"
echo "Subject std: $(fslstats ${SUBJECT_BRAIN} -S)"
echo "Template mean: $(fslstats ${MNI_TEMPLATE_BRAIN} -M)"
echo "Template std: $(fslstats ${MNI_TEMPLATE_BRAIN} -S)"

cmd="fslmaths ${SUBJECT_BRAIN} -sub $(fslstats ${SUBJECT_BRAIN} -M) -div $(fslstats ${SUBJECT_BRAIN} -S) -mul $(fslstats ${MNI_TEMPLATE_BRAIN} -S) -add $(fslstats ${MNI_TEMPLATE_BRAIN} -M) ${SUBJECT_BRAIN_NORMALIZED}"
echo $cmd

# Run the command and check for errors
$cmd
if [ $? -ne 0 ]; then
    echo "Error during normalization"
    exit 1
fi
