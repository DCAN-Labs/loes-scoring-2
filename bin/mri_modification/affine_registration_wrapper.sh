#!/bin/bash

module load fsl

# Example registration command: /panfs/roc/msisoft/fsl/6.0.4/bin/flirt -in /home/feczk001/shared/projects/S1067_Loes/data/Loes_DataLad/sub-1043EATO/ses-20190426/mprage.nii.gz \
# -ref /home/feczk001/shared/projects/S1067_Loes/data/MNI152/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii -out /home/feczk001/shared/projects/S1067_Loes/data/MNI-space_Loes_data/sub-1043EATO_ses-20190426_space_MNI_mprage.nii.gz -dof 9

MRI_IN=$1
MRI_OUT=$2
REF=$3

echo Registering ${MRI_IN}
cmd="flirt -in ${MRI_IN} -ref ${REF} -out ${MRI_OUT}"
echo $cmd
$cmd
#!/bin/bash

module load fsl

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <MRI_IN> <MRI_OUT> <REF>"
  exit 1
fi

echo "Registering ${MRI_IN} to ${REF}, output will be saved as ${MRI_OUT}"

# Constructing and running the flirt command
cmd="flirt -in ${MRI_IN} -ref ${REF} -out ${MRI_OUT}"
echo "Running command: $cmd"
$cmd

if [ $? -eq 0 ]; then
  echo "Registration successful."
else
  echo "Registration failed."
  exit 1
fi
