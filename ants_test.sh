# get an srun: srun --time=8:00:00 --mem=32GB -c 8 --tmp=20gb -p interactive -A feczk001 --x11 --pty bash

MRI_IN="/users/1/lundq163/ss_mprage.nii.gz"
MRI_OUT="/users/1/lundq163/reg_ss_mprage.nii.gz"
REF="/home/feczk001/shared/projects/S1067_Loes/data/MNI152/mni_icbm152_t1_tal_nlin_sym_09a_masked.nii.gz"

#module load ants
echo Registering ${MRI_IN}
cmd="antsRegistration --collapse-output-transforms 1 --dimensionality 3 --float 1 --initialize-transforms-per-stage 0 --interpolation LanczosWindowedSinc --output [  transform, ${MRI_OUT} ] --transform Rigid[ 0.05 ] --metric Mattes[ ${REF}, ${MRI_IN}, 1, 56, Regular, 0.25 ] --convergence [ 100x100, 1e-06, 20 ] --smoothing-sigmas 2.0x1.0vox --shrink-factors 2x1 --use-histogram-matching 1 --transform Affine[ 0.08 ] --metric Mattes[ ${REF}, ${MRI_IN}, 1, 56, Regular, 0.25 ] --convergence [ 100x100, 1e-06, 20 ] --smoothing-sigmas 1.0x0.0vox --shrink-factors 2x1 --use-histogram-matching 1 --transform SyN[ 0.1, 3.0, 0.0 ] --metric CC[ ${REF}, ${MRI_IN}, 1, 4, None, 1 ] --convergence [ 100x70x50x20, 1e-06, 10 ] --smoothing-sigmas 3.0x2.0x1.0x0.0vox --shrink-factors 8x4x2x1 --use-histogram-matching 1 --winsorize-image-intensities [ 0.005, 0.995 ]  --write-composite-transform 1 -v"
echo $cmd
$cmd