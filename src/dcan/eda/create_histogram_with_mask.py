import numpy as np
import nibabel as nib
from pathlib import Path


def load_nifti(file_path):
    """Load a NIfTI file and return the image and its data."""
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        return img, data
    except Exception as e:
        raise RuntimeError(f"Error loading NIfTI file: {file_path}. Error: {e}")

def apply_mask(image_data, mask_data):
    """Apply the mask to the image data."""
    return np.where(mask_data > 0, image_data, 0)

def save_nifti(data, affine, output_path):
    """Save the masked data as a new NIfTI file."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(data, affine), output_path)
    except Exception as e:
        raise RuntimeError(f"Error saving NIfTI file: {output_path}. Error: {e}")

def process_subject(subject_id, anatomical_type, in_dir, masked_img_dir):
    try:
        # Define paths
        mask_dir = Path(f'/home/feczk001/shared/projects/S1067_Loes/code/cortical_masking_work/anatonly_derivatives_nonGD/sub-0{subject_id}/ses-01/anat/')
        in_file = Path(f'sub-0{subject_id}_ses-01_space-MNI_brain_normalized_mprage.nii.gz')
        in_path = in_dir / in_file
        mask_file = f'sub-0{subject_id}_ses-01_run-01_space-MNI_label-{anatomical_type}_probseg.nii.gz'
        mask_path = mask_dir / mask_file
        masked_img_file = f'masked-sub-0{subject_id}_ses-01_space-MNI_brain_normalized_{anatomical_type}_mprage.nii.gz'

        # Load images and mask
        regular_img, regular_data = load_nifti(in_path)
        _, mask_data = load_nifti(mask_path)

        # Apply mask
        masked_data = apply_mask(regular_data, mask_data)

        # Save masked image
        save_nifti(masked_data, regular_img.affine, masked_img_dir / masked_img_file)
        print(f"Successfully processed subject {subject_id}, {anatomical_type}")
    
    except Exception as e:
        print(f"Error processing subject {subject_id}, {anatomical_type}: {e}")

def main():
    good_subjects = [1, 2, 4]  # List of good subjects
    in_dir = Path('/home/feczk001/shared/projects/S1067_Loes/data/niftis_deID/transformed')

    # Loop over subjects and anatomical types
    for subject_id in good_subjects:
        for anatomical_type in ["WM", "GM"]:
            masked_img_dir = Path(f'/home/feczk001/shared/projects/S1067_Loes/data/niftis_deID/masked/non_gd/{anatomical_type}/')
            process_subject(subject_id, anatomical_type, in_dir, masked_img_dir)

if __name__ == "__main__":
    main()