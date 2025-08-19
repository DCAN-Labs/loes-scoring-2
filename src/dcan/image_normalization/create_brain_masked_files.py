import sys
import numpy as np
import nibabel as nib
from pathlib import Path


def load_nifti(file_path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    """Load a NIfTI file and return the image object and its data array."""
    try:
        img = nib.load(str(file_path))
        return img, img.get_fdata()
    except Exception as e:
        raise RuntimeError(f"Error loading NIfTI file: {file_path}. Error: {e}")


def apply_mask(
    image_data: np.ndarray, mask_data: np.ndarray, mask_value: float = 1.0
) -> np.ndarray:
    """Apply a binary mask to the image data."""
    return np.where(np.isclose(mask_data, mask_value, rtol=1e-2), image_data, 0)


def save_nifti(data: np.ndarray, affine: np.ndarray, output_path: Path):
    """Save the NIfTI data to the specified output path."""
    try:
        output_path.parent.mkdir(
            parents=True, exist_ok=True
        )  # Ensure the directory exists
        nib.save(nib.Nifti1Image(data, affine), str(output_path))
    except Exception as e:
        raise RuntimeError(f"Error saving NIfTI file: {output_path}. Error: {e}")


def get_file_identifiers(file_name: str) -> tuple[str, str, str]:
    """
    Extract subject, session, and run identifiers from the file name.
    Assumes a specific naming convention: `sub-XXXXXX_ses-YYYYYY`.
    """
    try:
        parts = file_name.split("_")
        subject_id = next(part for part in parts if part.startswith("sub-"))
        session_id = next(part for part in parts if part.startswith("ses-"))
        run_id = next(
            (part for part in parts if part.startswith("run-")), "run-00"
        )  # Default to "run-00"
        return subject_id, session_id, run_id
    except Exception as e:
        raise ValueError(
            f"Error extracting identifiers from file name: {file_name}. Error: {e}"
        )


def validate_file_path(file_path: Path, file_description: str):
    """Ensure the file path exists."""
    if not file_path.exists():
        raise FileNotFoundError(f"{file_description} not found: {file_path}")


def process_subject(in_file: Path, mask_path: Path, masked_img_dir: Path):
    """Process a single subject by applying a mask and saving the result."""
    try:
        # Extract identifiers from the file name
        subject_id, session_id, run_id = get_file_identifiers(in_file.name)
        id_str = f"{subject_id}_{session_id}_{run_id}"

        masked_img_path = masked_img_dir / f"{id_str}_space-MNI_mprage.nii.gz"

        # Validate input and mask paths
        validate_file_path(in_file, "Input image")
        validate_file_path(mask_path, "Mask file")

        # Load input image and mask
        img, img_data = load_nifti(in_file)
        _, mask_data = load_nifti(mask_path)

        # Apply the mask and save the result
        masked_data = apply_mask(img_data, mask_data)
        save_nifti(masked_data, img.affine, masked_img_path)

        print(f"Successfully processed: {subject_id}, session: {session_id}")

    except Exception as e:
        print(f"Error processing subject {in_file.name}: {e}")


def process_directory(input_dir: Path, mask_path: Path, output_dir: Path):
    """Process all NIfTI files in the input directory."""
    try:
        # Validate input directory and mask file
        validate_file_path(input_dir, "Input directory")
        validate_file_path(mask_path, "Mask file")

        for in_file in input_dir.glob("*.nii.gz"):
            # Skip files containing 'Gd' in their names
            if "Gd" in in_file.name:
                continue

            process_subject(in_file, mask_path, output_dir)

    except Exception as e:
        print(f"Error processing directory {input_dir}: {e}")
        sys.exit(1)


def main(input_dir: str, mask_file: str, output_dir: str):
    """Main function to set up directories and start processing."""
    input_dir = Path(input_dir)
    mask_file = Path(mask_file)
    output_dir = Path(output_dir)

    try:
        process_directory(input_dir, mask_file, output_dir)
    except Exception as e:
        print(f"Critical error encountered: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: python script.py <input_directory> <mask_file> <output_directory>"
        )
        sys.exit(1)

    input_dir, mask_file, output_dir = sys.argv[1:4]

    main(input_dir, mask_file, output_dir)
