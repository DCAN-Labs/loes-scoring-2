import os
import sys
import numpy as np
import nibabel as nib
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import nilearn.image
import tqdm
import matplotlib

# Set up Matplotlib backend
matplotlib.use('TkAgg', force=True)
print("Switched to:", matplotlib.get_backend())

EPSILON = 0.01  # Threshold to filter out near-zero values


def get_file_identifiers(file: Path) -> tuple[str, str, str]:
    """
    Extract subject, session, and run identifiers from the file name.
    Assumes a specific naming convention: `sub-XXXXXX_ses-YYYYYY_run-ZZ_space-...`.
    """
    try:
        parts = file.stem.split("_")
        subject_id = next(part for part in parts if part.startswith("subject-"))
        session_id = next(part for part in parts if part.startswith("session-"))
        run_id = next((part for part in parts if part.startswith("run-")), "run-01")  # Default run identifier
        return subject_id, session_id, run_id
    except StopIteration as e:
        raise ValueError(f"Invalid file name format: {file.name}. Error: {e}")


def load_nifti(file_path: Path) -> np.ndarray:
    """
    Load and return the NIfTI image data from the specified path.
    """
    try:
        img = nib.load(str(file_path))
        return img.get_fdata()
    except Exception as e:
        print(f"Error loading NIfTI image at {file_path}: {e}")
        return None


def mask_in_matter_data(img_dir: Path, subject_id: str, session_id: str, mask_path: Path) -> np.ndarray:
    """
    Apply a mask to the subject's brain image and return the filtered voxel intensities.
    """
    brain_path = img_dir / f"{subject_id}_{session_id}_run-00_space-MNI_mprage_RAVEL.nii.gz"

    if not brain_path.exists():
        print(f"Missing brain image: {brain_path}. Skipping subject {subject_id}-{session_id}.")
        return None

    if not mask_path.exists():
        print(f"Missing mask file: {mask_path}. Skipping subject {subject_id}-{session_id}.")
        return None

    try:
        brain_data = load_nifti(brain_path)
        mask_data = load_nifti(mask_path)

        if brain_data is None or mask_data is None:
            return None

        masked_brain_data = brain_data * (mask_data == 1)
        return masked_brain_data[masked_brain_data > EPSILON]  # Remove near-zero values

    except Exception as e:
        print(f"Error processing mask on {brain_path}: {e}")
        return None


def plot_voxel_intensity_histogram(
    subject_id: str, session_id: str, img_dir: Path, hist_dir: Path, wm_mask: Path, gm_mask: Path
):
    """
    Generate and save a histogram of voxel intensities for a subject's gray and white matter.
    """
    gm_data = mask_in_matter_data(img_dir, subject_id, session_id, gm_mask)
    wm_data = mask_in_matter_data(img_dir, subject_id, session_id, wm_mask)

    if gm_data is None or wm_data is None:
        print(f"Skipping histogram generation for {subject_id}-{session_id} due to missing data.")
        return

    # Plot histograms
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.set_xlim([0, 600])

    sns.histplot(gm_data.flatten(), label="Gray Matter (GM)", kde=True, color="blue", alpha=0.6)
    sns.histplot(wm_data.flatten(), label="White Matter (WM)", kde=True, color="orange", alpha=0.6)

    plt.title(f"Voxel Intensity Distribution for {subject_id}-{session_id}")
    plt.xlabel("Voxel Intensity")
    plt.ylabel("Frequency")
    plt.legend()

    # Save histogram
    hist_path = hist_dir / f"masked-{subject_id}_{session_id}_space-MNI_brain_voxel_histogram.png"
    hist_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(hist_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved histogram for {subject_id}-{session_id} at {hist_path}")


def process_directory(img_dir: Path, hist_dir: Path, wm_mask: Path, gm_mask: Path):
    """
    Process all subjects in the image directory and generate histograms.
    """
    img_dir = Path(img_dir)
    hist_dir = Path(hist_dir)

    # List all subject files with _RAVEL.nii.gz suffix
    files = sorted(img_dir.glob("*_RAVEL.nii.gz"))

    if not files:
        print(f"No matching NIfTI files found in {img_dir}. Check input directory.")
        sys.exit(1)

    for file in tqdm.tqdm(files, desc="Processing Subjects"):
        try:
            subject_id, session_id, _ = get_file_identifiers(file)
            plot_voxel_intensity_histogram(subject_id, session_id, img_dir, hist_dir, wm_mask, gm_mask)
        except ValueError as e:
            print(f"Skipping file {file.name}: {e}")
            continue


def main(img_dir: str, hist_dir: str, wm_mask: str, gm_mask: str):
    """
    Main function to process all subjects and generate histograms.
    """
    img_dir = Path(img_dir)
    hist_dir = Path(hist_dir)
    wm_mask = Path(wm_mask)
    gm_mask = Path(gm_mask)

    # Validate directories and mask files
    if not img_dir.exists():
        print(f"Error: Input directory does not exist: {img_dir}")
        sys.exit(1)

    if not wm_mask.exists() or not gm_mask.exists():
        print(f"Error: One or both mask files are missing.\nWM Mask: {wm_mask}\nGM Mask: {gm_mask}")
        sys.exit(1)

    process_directory(img_dir, hist_dir, wm_mask, gm_mask)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python script.py <image_directory> <histogram_output_directory> <wm_mask> <gm_mask>")
        sys.exit(1)

    img_dir = sys.argv[1]
    hist_dir = sys.argv[2]
    wm_mask = sys.argv[3]
    gm_mask = sys.argv[4]

    try:
        main(img_dir, hist_dir, wm_mask, gm_mask)
    except Exception as e:
        print(f"Critical error encountered: {e}")
        sys.exit(1)
