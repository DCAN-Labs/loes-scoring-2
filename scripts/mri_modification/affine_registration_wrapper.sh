#!/bin/bash

# Script for registering MRI images using ANTs
# Author: Paul Reiners, Jacob Lundquist
# Date: 12/10/2024

# Load necessary modules
module load ants

# Ensure correct usage
if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <MRI_IN> <MRI_OUT> <REF> <REG>"
  echo "  MRI_IN: Input MRI file to be registered"
  echo "  MRI_OUT: Output registered MRI file"
  echo "  REF: Reference MRI file"
  echo "  REG: Prefix for transformation outputs"
  exit 1
fi

# Parse command-line arguments
MRI_IN=$1
MRI_OUT=$2
REF=$3
REG=$4

echo "Registering ${MRI_IN} to ${REF}"
echo "Output will be saved as ${MRI_OUT}, with transformations saved as ${REG}"

# Construct the antsRegistration command
cmd=$(cat <<EOF
antsRegistration --collapse-output-transforms 1 \
  --dimensionality 3 \
  --float 1 \
  --initialize-transforms-per-stage 0 \
  --interpolation LanczosWindowedSinc \
  --output [${REG}, ${MRI_OUT}] \
  --transform Rigid[0.05] \
  --metric Mattes[${REF}, ${MRI_IN}, 1, 56, Regular, 0.25] \
  --convergence [100x100, 1e-06, 20] \
  --smoothing-sigmas 2.0x1.0vox \
  --shrink-factors 2x1 \
  --use-histogram-matching 1 \
  --transform Affine[0.08] \
  --metric Mattes[${REF}, ${MRI_IN}, 1, 56, Regular, 0.25] \
  --convergence [100x100, 1e-06, 20] \
  --smoothing-sigmas 1.0x0.0vox \
  --shrink-factors 2x1 \
  --use-histogram-matching 1 \
  --transform SyN[0.1, 3.0, 0.0] \
  --metric CC[${REF}, ${MRI_IN}, 1, 4, None, 1] \
  --convergence [100x70x50x20, 1e-06, 10] \
  --smoothing-sigmas 3.0x2.0x1.0x0.0vox \
  --shrink-factors 8x4x2x1 \
  --use-histogram-matching 1 \
  --winsorize-image-intensities [0.005, 0.995] \
  --write-composite-transform 1 \
  -v
EOF
)

# Print and execute the command
echo "Running command:"
echo "${cmd}"
eval "${cmd}"

# Check the command's success
if [ $? -eq 0 ]; then
  echo "Registration completed successfully."
else
  echo "Registration failed. Please check the inputs and try again."
  exit 1
fi
