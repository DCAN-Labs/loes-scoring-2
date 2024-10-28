import sys
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

def apply_mask(image_data, mask_data, matter_type):
    """Apply the mask to the image data based on the specified matter type."""
    matter_type_dict = {'WM': 2, 'GM': 3}  # Mapping matter type to integer
    mask_value = matter_type_dict.get(matter_type, 0)
    return np.where(mask_data == mask_value, image_data, 0)

def save_nifti(data, affine, output_path):
    """Save the masked data as a new NIfTI file."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        nib.save(nib.Nifti1Image(data, affine), output_path)
    except Exception as e:
        raise RuntimeError(f"Error saving NIfTI file: {output_path}. Error: {e}")

def get_file_identifiers(file_name):
    """Extract subject, session, and run identifiers from the file name."""
    return file_name[:6], file_name[7:13], 'run-01'

def process_subject(in_file, anatomical_type, in_dir, mask_dir, masked_img_dir):
    subject_id, session_id, run_id = get_file_identifiers(in_file)
    id_str = f"{subject_id}_{session_id}_{run_id}"
    
    in_path = in_dir / in_file
    mask_path = mask_dir / f"{id_str}_space-MNI_dseg.nii.gz"
    masked_img_path = masked_img_dir / f"{id_str}_space-MNI_{anatomical_type}_mprage.nii.gz"

    if not in_path.exists() or not mask_path.exists():
        print(f"File missing for subject {subject_id} ({'image' if not in_path.exists() else 'mask'}): Skipping.")
        return

    try:
        # Load images and mask
        img, img_data = load_nifti(in_path)
        _, mask_data = load_nifti(mask_path)

        # Apply mask and save
        masked_data = apply_mask(img_data, mask_data, anatomical_type)
        save_nifti(masked_data, img.affine, masked_img_path)
        print(f"Successfully processed {subject_id}, anatomical type: {anatomical_type}")

    except Exception as e:
        print(f"Error processing {subject_id} with anatomical type {anatomical_type}: {e}")

def main(in_dir, mask_dir, masked_img_dir):
    in_dir, mask_dir, masked_img_dir = Path(in_dir), Path(mask_dir), Path(masked_img_dir)
    # Process each subject file
    for in_file in in_dir.glob("*.nii.gz"):
        if 'Gd' in in_file.name:
            continue
        for anatomical_type in ["WM", "GM"]:
            process_subject(in_file.name, anatomical_type, in_dir, mask_dir, masked_img_dir)

if __name__ == "__main__":
    # Sample arguments
    in_dir = sys.argv[1]
    mask_dir = sys.argv[2]
    masked_img_dir = sys.argv[3]
    main(in_dir, mask_dir, masked_img_dir)
