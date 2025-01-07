import seaborn as sns
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
import sys


def get_file_identifiers(file_name):
    """
    Extract subject, session, and run identifiers from the file name.
    Assumes a specific naming convention.
    """
    try:
        subject_id = file_name[:6]
        session_id = file_name[7:13]
        run_id = "run-01"  # Default run identifier
        return subject_id, session_id, run_id
    except IndexError as e:
        raise ValueError(f"Invalid file name format: {file_name}. Error: {e}")


def get_data(img_path):
    """
    Load and flatten the NIfTI image at the specified path, filtering out zero intensities.
    """
    try:
        img = nib.load(img_path)
        data = img.get_fdata().flatten()
        return data[data > 0]  # Filter out zero intensities
    except Exception as e:
        print(f"Error loading NIfTI image at {img_path}: {e}")
        return None


def plot_voxel_intensity_histogram(subject_id, session_id, img_dir, hist_dir, gm_file, wm_file):
    """
    Generate and save a histogram of voxel intensities for a subject's brain images.
    """
    gm_img_path = img_dir / gm_file
    wm_img_path = img_dir / wm_file

    # Load data for GM and WM images
    gm_data = get_data(gm_img_path)
    wm_data = get_data(wm_img_path)

    if gm_data is None or wm_data is None:
        print(f"Data missing for subject {subject_id}-{session_id}. Skipping histogram generation.")
        return

    # Plot histograms
    plt.figure(figsize=(10, 6))
    sns.histplot(gm_data, label="GM", kde=True, color="blue", alpha=0.6)
    sns.histplot(wm_data, label="WM", kde=True, color="orange", alpha=0.6)
    plt.title(f"Distribution of Voxel Intensities for {subject_id}-{session_id}")
    plt.xlabel("Voxel Intensity")
    plt.ylabel("Frequency")
    plt.legend()

    # Save histogram plot
    hist_path = hist_dir / f"masked-sub-{subject_id}_{session_id}_space-MNI_brain_voxel_histogram.png"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(hist_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Histogram saved for subject {subject_id}-{session_id} at {hist_path}")


def main(img_dir, hist_dir):
    """
    Main function to process all files and generate histograms for each subject.
    """
    img_dir = Path(img_dir)
    hist_dir = Path(hist_dir)

    # List all NIfTI files in the image directory
    files = sorted(f for f in img_dir.glob("*.nii.gz"))

    if len(files) % 2 != 0:
        print("Warning: Unpaired GM and WM files found. Ensure files are correctly paired.")

    # Process files in pairs (GM and WM)
    for i in range(0, len(files), 2):
        try:
            gm_file = files[i].name
            wm_file = files[i + 1].name
            subject_id, session_id, _ = get_file_identifiers(gm_file)
            plot_voxel_intensity_histogram(subject_id, session_id, img_dir, hist_dir, gm_file, wm_file)
        except IndexError:
            print(f"Error: Unpaired file at index {i}. Skipping.")
            continue
        except ValueError as e:
            print(f"Error processing file: {e}")
            continue


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <image_directory> <histogram_output_directory>")
        sys.exit(1)

    img_dir_in = sys.argv[1]
    hist_dir_out = sys.argv[2]

    try:
        main(img_dir_in, hist_dir_out)
    except Exception as e:
        print(f"Critical error: {e}")
        sys.exit(1)
