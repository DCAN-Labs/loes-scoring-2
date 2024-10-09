#!/bin/bash

module load fsl

SUBJECT_BRAIN=$1
MNI_TEMPLATE_BRAIN=$2
SUBJECT_BRAIN_NORMALIZED=$3

echo Normalizing ${SUBJECT_BRAIN}
cmd="fslmaths ${SUBJECT_BRAIN} -sub `fslstats ${SUBJECT_BRAIN} -M` -div `fslstats ${SUBJECT_BRAIN} -S` -mul `fslstats ${MNI_TEMPLATE_BRAIN} -S` -add `fslstats ${MNI_TEMPLATE_BRAIN} -M` ${SUBJECT_BRAIN_NORMALIZED}"
echo $cmd
$cmd
