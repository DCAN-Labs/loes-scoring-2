import sys
from pathlib import Path
import numpy as np
import nibabel as nib
from tqdm import tqdm
from dcan.image_normalization.create_brain_masked_files import get_file_identifiers


def load_nifti(file_path: Path):
    """Load a NIfTI file and return the image object and its data array."""
    try:
        img = nib.load(str(file_path))
        return img, img.get_fdata()
    except Exception as e:
        raise RuntimeError(f"Error loading NIfTI file at {file_path}: {e}")


def apply_mask(
    image_data: np.ndarray, mask_data: np.ndarray, mask_value: float = 1.0
) -> np.ndarray:
    """Apply a binary mask to image data, setting voxels outside the mask to 0."""
    return np.where(np.isclose(mask_data, mask_value, rtol=1e-2), image_data, 0)


def save_nifti(data: np.ndarray, affine: np.ndarray, output_path: Path):
    """Save the masked data to a new NIfTI file."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(data, affine), str(output_path))
    except Exception as e:
        raise RuntimeError(f"Error saving NIfTI file to {output_path}: {e}")


def process_subject(
    in_file: Path, anatomical_type: str, in_dir: Path, mask_path: Path, output_dir: Path
):
    """Process a single subject by applying a mask and saving the masked image."""
    subject_id, session_id, run_id = get_file_identifiers(in_file.name)
    id_str = f"{subject_id}_{session_id}_{run_id}"
    in_path = in_dir / in_file
    output_file = output_dir / f"{id_str}_space-MNI_{anatomical_type}_mprage.nii.gz"

    if not in_path.exists():
        print(
            f"Input image file missing for subject-session {subject_id}-{session_id}: Skipping."
        )
        return

    if not mask_path.exists():
        print(
            f"Mask file missing: {mask_path}. Skipping subject {subject_id}-{session_id}."
        )
        return

    try:
        # Load input image and mask
        img, img_data = load_nifti(in_path)
        _, mask_data = load_nifti(mask_path)

        # Apply mask and save result
        masked_data = apply_mask(img_data, mask_data)
        save_nifti(masked_data, img.affine, output_file)

        print(
            f"Processed {subject_id}, session {session_id}, anatomical type: {anatomical_type}"
        )

    except Exception as e:
        print(f"Error processing {in_file.name} ({anatomical_type}): {e}")


def main(in_dir: str, mask_file: str, masked_img_dir: str):
    """Main function to process all subject files in the input directory."""
    in_dir = Path(in_dir)
    mask_file = Path(mask_file)
    masked_img_dir = Path(masked_img_dir)

    # Iterate through all .nii.gz files in the input directory, excluding files with 'Gd'
    for in_file in tqdm(in_dir.glob("*.nii.gz"), desc="Processing files"):
        if "Gd" in in_file.name:
            continue
        process_subject(in_file, "CSF", in_dir, mask_file, masked_img_dir)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: python script.py <input_directory> <mask_file> <output_directory>"
        )
        sys.exit(1)

    input_dir, mask_file, output_dir = sys.argv[1:4]

    try:
        main(input_dir, mask_file, output_dir)
    except Exception as e:
        print(f"Critical error: {e}")
        sys.exit(1)
