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


def apply_mask(image_data, mask_data, mask_value=1):
    """Apply the mask to the image data."""
    return np.where(np.isclose(mask_data, mask_value, rtol=1e-2), image_data, 0)


def save_nifti(data, affine, output_path):
    """Save the masked data as a new NIfTI file."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        nib.save(nib.Nifti1Image(data, affine), output_path)
    except Exception as e:
        raise RuntimeError(f"Error saving NIfTI file: {output_path}. Error: {e}")


def get_file_identifiers(file_name):
    """
    Extract subject, session, and run identifiers from the file name.
    Assumes specific naming convention: `sub-XXXXXX_ses-YYYYYY`.
    """
    try:
        subject_id = file_name[:6]
        session_id = file_name[7:13]
        run_id = "run-01"  # Default run identifier
        return subject_id, session_id, run_id
    except Exception as e:
        raise ValueError(f"Error extracting identifiers from file name: {file_name}. Error: {e}")


def process_subject(in_file, anatomical_type, in_dir, mask_dir, masked_img_dir):
    """Process a single subject by applying the mask and saving the result."""
    try:
        subject_id, session_id, run_id = get_file_identifiers(in_file.name)
        id_str = f"{subject_id}_{session_id}_{run_id}"

        in_path = in_dir / in_file
        mask_path = mask_dir / f"mni_icbm152_{anatomical_type.lower()}_tal_nlin_sym_09a_int_rounded.nii"
        masked_img_path = masked_img_dir / f"{id_str}_space-MNI_{anatomical_type}_mprage.nii.gz"

        if not in_path.exists():
            print(f"Image file missing for subject {subject_id}: Skipping.")
            return

        if not mask_path.exists():
            print(f"Mask file missing for {anatomical_type}: Skipping subject {subject_id}.")
            return

        # Load images and mask
        img, img_data = load_nifti(in_path)
        _, mask_data = load_nifti(mask_path)

        # Apply mask and save
        masked_data = apply_mask(img_data, mask_data)
        save_nifti(masked_data, img.affine, masked_img_path)
        print(f"Successfully processed {subject_id}, anatomical type: {anatomical_type}")

    except Exception as e:
        print(f"Error processing subject {in_file.name} ({anatomical_type}): {e}")


def main(in_dir, mask_dir, masked_img_dir):
    """Main function to process all subject files."""
    in_dir, mask_dir, masked_img_dir = Path(in_dir), Path(mask_dir), Path(masked_img_dir)

    # Iterate through all .nii.gz files in the input directory
    for in_file in in_dir.glob("*.nii.gz"):
        # Skip files containing 'Gd' in their name
        if 'Gd' in in_file.name:
            continue

        # Process for both White Matter (WM) and Gray Matter (GM)
        for anatomical_type in ["WM", "GM"]:
            process_subject(in_file, anatomical_type, in_dir, mask_dir, masked_img_dir)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <input_directory> <mask_directory> <output_directory>")
        sys.exit(1)

    input_dir = sys.argv[1]
    mask_dir = sys.argv[2]
    output_dir = sys.argv[3]

    try:
        main(input_dir, mask_dir, output_dir)
    except Exception as e:
        print(f"Critical error: {e}")
        sys.exit(1)
