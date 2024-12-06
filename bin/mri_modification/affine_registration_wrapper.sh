#!/bin/bash

module load ants

# Example registration command: 
#antsRegistration --collapse-output-transforms 1 --dimensionality 3 --float 1 --initialize-transforms-per-stage 0 \
# --interpolation LanczosWindowedSinc --output [ /users/1/lundq163/transform, /users/1/lundq163/reg_ss_mprage.nii.gz ] --transform Rigid[ 0.05 ] \
# --metric Mattes[ /home/feczk001/shared/projects/S1067_Loes/data/MNI152/mni_icbm152_t1_tal_nlin_sym_09a_masked.nii.gz, /users/1/lundq163/ss_mprage.nii.gz, 1, 56, Regular, 0.25 ] \
# --convergence [ 100x100, 1e-06, 20 ] --smoothing-sigmas 2.0x1.0vox --shrink-factors 2x1 --use-histogram-matching 1 --transform Affine[ 0.08 ] \
# --metric Mattes[ /home/feczk001/shared/projects/S1067_Loes/data/MNI152/mni_icbm152_t1_tal_nlin_sym_09a_masked.nii.gz, /users/1/lundq163/ss_mprage.nii.gz, 1, 56, Regular, 0.25 ] \
# --convergence [ 100x100, 1e-06, 20 ] --smoothing-sigmas 1.0x0.0vox --shrink-factors 2x1 --use-histogram-matching 1 --transform SyN[ 0.1, 3.0, 0.0 ] \
# --metric CC[ /home/feczk001/shared/projects/S1067_Loes/data/MNI152/mni_icbm152_t1_tal_nlin_sym_09a_masked.nii.gz, /users/1/lundq163/ss_mprage.nii.gz, 1, 4, None, 1 ] \
# --convergence [ 100x70x50x20, 1e-06, 10 ] --smoothing-sigmas 3.0x2.0x1.0x0.0vox --shrink-factors 8x4x2x1 --use-histogram-matching 1 \
# --winsorize-image-intensities [ 0.005, 0.995 ] --write-composite-transform 1 -v

# note that you may need to get an srun. here is example: srun --time=8:00:00 --mem=32GB -c 8 --tmp=20gb -p interactive -A feczk001 --x11 --pty bash

MRI_IN=$1
MRI_OUT=$2
REF=$3
REG=$4

echo Registering ${MRI_IN}
cmd="antsRegistration --collapse-output-transforms 1 --dimensionality 3 --float 1 --initialize-transforms-per-stage 0 --interpolation LanczosWindowedSinc --output [  ${REG}, ${MRI_OUT} ] --transform Rigid[ 0.05 ] --metric Mattes[ ${REF}, ${MRI_IN}, 1, 56, Regular, 0.25 ] --convergence [ 100x100, 1e-06, 20 ] --smoothing-sigmas 2.0x1.0vox --shrink-factors 2x1 --use-histogram-matching 1 --transform Affine[ 0.08 ] --metric Mattes[ ${REF}, ${MRI_IN}, 1, 56, Regular, 0.25 ] --convergence [ 100x100, 1e-06, 20 ] --smoothing-sigmas 1.0x0.0vox --shrink-factors 2x1 --use-histogram-matching 1 --transform SyN[ 0.1, 3.0, 0.0 ] --metric CC[ ${REF}, ${MRI_IN}, 1, 4, None, 1 ] --convergence [ 100x70x50x20, 1e-06, 10 ] --smoothing-sigmas 3.0x2.0x1.0x0.0vox --shrink-factors 8x4x2x1 --use-histogram-matching 1 --winsorize-image-intensities [ 0.005, 0.995 ]  --write-composite-transform 1 -v"
echo $cmd
$cmd
#!/bin/bash

module load ants

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <MRI_IN> <MRI_OUT> <REF>"
  exit 1
fi

echo "Registering ${MRI_IN} to ${REF}, output will be saved as ${MRI_OUT} and transfrom as ${REG}"

# Constructing and running the antsRegistration command
cmd="antsRegistration --collapse-output-transforms 1 --dimensionality 3 --float 1 --initialize-transforms-per-stage 0 --interpolation LanczosWindowedSinc --output [  ${REG}, ${MRI_OUT} ] --transform Rigid[ 0.05 ] --metric Mattes[ ${REF}, ${MRI_IN}, 1, 56, Regular, 0.25 ] --convergence [ 100x100, 1e-06, 20 ] --smoothing-sigmas 2.0x1.0vox --shrink-factors 2x1 --use-histogram-matching 1 --transform Affine[ 0.08 ] --metric Mattes[ ${REF}, ${MRI_IN}, 1, 56, Regular, 0.25 ] --convergence [ 100x100, 1e-06, 20 ] --smoothing-sigmas 1.0x0.0vox --shrink-factors 2x1 --use-histogram-matching 1 --transform SyN[ 0.1, 3.0, 0.0 ] --metric CC[ ${REF}, ${MRI_IN}, 1, 4, None, 1 ] --convergence [ 100x70x50x20, 1e-06, 10 ] --smoothing-sigmas 3.0x2.0x1.0x0.0vox --shrink-factors 8x4x2x1 --use-histogram-matching 1 --winsorize-image-intensities [ 0.005, 0.995 ]  --write-composite-transform 1 -v"
echo "Running command: $cmd"
$cmd

if [ $? -eq 0 ]; then
  echo "Registration successful."
else
  echo "Registration failed."
  exit 1
fi
