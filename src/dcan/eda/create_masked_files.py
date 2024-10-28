import sys
import numpy as np
import nibabel as nib
from pathlib import Path
from os import listdir
from os.path import isfile, join

def load_nifti(file_path):
    """Load a NIfTI file and return the image and its data."""
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        return img, data
    except Exception as e:
        raise RuntimeError(f"Error loading NIfTI file: {file_path}. Error: {e}")

def apply_mask(image_data, mask_data, matter_type):
    """Apply the mask to the image data."""
    matter_type_int = {'WM': 2, 'GM': 3}.get(matter_type, 0)  # Use a dictionary for clarity
    return np.where(mask_data.astype(int) == matter_type_int, image_data, 0)

def save_nifti(data, affine, output_path):
    """Save the masked data as a new NIfTI file."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        nib.save(nib.Nifti1Image(data, affine), output_path)
    except Exception as e:
        raise RuntimeError(f"Error saving NIfTI file: {output_path}. Error: {e}")

def process_subject(in_file, anatomical_type, in_dir, mask_dir, masked_img_dir):
    try:
        # Define paths
        in_path = in_dir / in_file
        formatted_subject_id = in_file[:6]
        formatted_session_id = in_file[7:13]
        formatted_run_id = 'run-01'
        id_str = f'{formatted_subject_id}_{formatted_session_id}_{formatted_run_id}'
        mask_file = f'{id_str}_space-MNI_dseg.nii.gz'
        mask_path = mask_dir / mask_file
        masked_img_file = f'{id_str}_space-MNI_{anatomical_type}_mprage.nii.gz'

        # Check if input files exist
        if not in_path.exists():
            raise FileNotFoundError(f"Input image not found: {in_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        # Load images and mask
        regular_img, regular_data = load_nifti(in_path)
        _, mask_data = load_nifti(mask_path)

        # Apply mask
        masked_data = apply_mask(regular_data, mask_data, anatomical_type)

        # Save masked image
        save_nifti(masked_data, regular_img.affine, masked_img_dir / masked_img_file)
        print(f"Successfully processed subject {formatted_subject_id}, {anatomical_type}")
    
    except Exception as e:
        print(f"Error processing subject {formatted_subject_id}, {anatomical_type}: {e}")

def main(in_dir, mask_dir, masked_img_dir):
    # Loop over subjects and anatomical types
    only_files = [f for f in listdir(in_dir) if isfile(join(in_dir, f))]
    for in_file in only_files:
        if in_file.endswith('Gd.nii.gz'):
            continue
        for anatomical_type in ["WM", "GM"]:
            process_subject(in_file, anatomical_type, in_dir, mask_dir, masked_img_dir)

if __name__ == "__main__":
    # Sample arguments
    # in_dir = Path('/home/feczk001/shared/projects/S1067_Loes/data/Fairview-ag_in_dn/ModelDev/June2023/Loes_score/anonymized_Loes_score/processed/')
    # mask_dir = Path(f'/home/feczk001/shared/projects/S1067_Loes/code/cortical_masking_work/Fairview-ag_nonGD_masks_MNI/')
    # masked_img_dir = Path('/home/feczk001/shared/projects/S1067_Loes/data/Fairview-ag_in_dn/ModelDev/June2023/Loes_score/anonymized_Loes_score/masked/')
    
    in_dir = Path(sys.argv[1])
    mask_dir = Path(sys.argv[2])
    masked_img_dir = Path(sys.argv[2])
    main(in_dir, mask_dir, masked_img_dir)
