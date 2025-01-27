import sys
from pathlib import Path
import numpy as np
import nibabel as nib
import pandas as pd
from typing import Tuple


def load_nifti(file_path: Path) -> Tuple[nib.Nifti1Image, np.ndarray]:
    """
    Load a NIfTI file and return the image object and its data as a NumPy array.
    """
    try:
        img = nib.load(str(file_path))
        data = img.get_fdata()
        return img, data
    except Exception as e:
        raise RuntimeError(f"Error loading NIfTI file {file_path}: {e}")


def apply_mask(image_data: np.ndarray, mask_data: np.ndarray, mask_value: float = 1.0) -> np.ndarray:
    """
    Apply a binary mask to the image data. Pixels matching `mask_value` in the mask are retained.
    """
    return np.where(np.isclose(mask_data, mask_value, rtol=1e-2), image_data, 0)


def save_nifti(data: np.ndarray, affine: np.ndarray, output_path: Path) -> None:
    """
    Save the masked data as a new NIfTI file.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(data, affine), str(output_path))
    except Exception as e:
        raise RuntimeError(f"Error saving NIfTI file {output_path}: {e}")


def file_exists(file_path: Path) -> bool:
    """
    Check if a file exists and log a warning if not.
    """
    if not file_path.exists():
        print(f"Warning: File not found: {file_path}")
        return False
    return True


def process_subject(row: pd.Series, in_file: str, in_dir: Path, mask_file: Path, output_dir: Path) -> None:
    """
    Process a single subject by applying a mask to their image data and saving the result.
    """
    subject_id = row['anonymized_subject_id']
    session_id = row['anonymized_session_id']
    input_file = f"{subject_id}_{session_id}_space-MNI_brain_{in_file}"

    in_path = in_dir / input_file
    if not file_exists(in_path) or not file_exists(mask_file):
        return

    try:
        # Load image and mask
        img, img_data = load_nifti(in_path)
        _, mask_data = load_nifti(mask_file)

        # Apply mask
        masked_data = apply_mask(img_data, mask_data)

        # Save output
        output_path = output_dir / input_file
        save_nifti(masked_data, img.affine, output_path)
        print(f"Processed: {subject_id}_{session_id}")
    except Exception as e:
        print(f"Error processing {subject_id}_{session_id}: {e}")


def mask_file(row: pd.Series, in_dir: Path, mask_file: Path, output_dir: Path) -> int:
    """
    Apply the mask to the subject's image file if eligible.
    Skips files containing 'Gd' in their name.
    """
    in_file = row['scan']
    if 'Gd' in in_file:
        return 0

    process_subject(row, in_file, in_dir, mask_file, output_dir)
    return 1


def main(input_dir: str, csv_file: str, mask_dir: str, output_dir: str) -> None:
    """
    Main function to process all subjects using data from the CSV file.
    """
    input_dir = Path(input_dir)
    csv_file = Path(csv_file)
    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)

    mask_file = mask_dir / "mni_icbm152_t1_tal_nlin_sym_09a_mask_int16.nii.gz"
    if not file_exists(mask_file):
        print(f"Critical: Mask file not found: {mask_file}")
        sys.exit(1)

    try:
        df = pd.read_csv(csv_file)
        df['masked'] = df.apply(mask_file, axis=1, args=(input_dir, mask_file, output_dir))
    except Exception as e:
        print(f"Critical error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python script.py <input_dir> <csv_file> <mask_directory> <output_directory>")
        sys.exit(1)

    input_dir, csv_file, mask_dir, output_dir = sys.argv[1:5]
    main(input_dir, csv_file, mask_dir, output_dir)
