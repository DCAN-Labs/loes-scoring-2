import sys
import os
import nibabel as nib
import numpy as np

def load_nifti(file_path):
    """Load a NIfTI file and return the loaded object and its data as a numpy array."""
    try:
        nifti_file = nib.load(file_path)
        data = nifti_file.get_fdata()
        return nifti_file, data
    except Exception as e:
        raise RuntimeError(f"Error loading NIfTI file at {file_path}: {e}")
    

def process_nifti_data(data, threshold):
    """Round the NIfTI data and convert it to integer type."""
    try:
        if np.isclose(threshold, 0.5):
            rounded_data = np.round(data)
        else:
            def threshhold_round(x):
                if np.ceil(x) - x < 1.0 - threshold:
                    return np.ceil(x)
                else:
                    return np.floor(x)
            vfunc = np.vectorize(threshhold_round)
            rounded_data = vfunc(data)
        int_data = rounded_data.astype(np.int16)
        
        return int_data
    except Exception as e:
        raise RuntimeError(f"Error processing NIfTI data: {e}")

def save_nifti(output_path, data, affine, header):
    """Save the processed NIfTI data to a file."""
    try:
        new_nifti = nib.Nifti1Image(data, affine, header)
        nib.save(new_nifti, output_path)
    except Exception as e:
        raise RuntimeError(f"Error saving NIfTI file to {output_path}: {e}")

def main(directory, input_nifti_file, output_nifti_file, threshold=0.5):
    """Main function to load, process, and save a NIfTI file."""
    input_path = os.path.join(directory, input_nifti_file)
    output_path = os.path.join(directory, output_nifti_file)

    # Load the NIfTI file
    nifti_file, data = load_nifti(input_path)

    # Process the NIfTI data
    rounded_data = process_nifti_data(data, threshold)

    # Save the processed NIfTI data to a new file
    save_nifti(output_path, rounded_data, nifti_file.affine, nifti_file.header)
    print(f"Processed NIfTI file saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4 and len(sys.argv) != 5:
        print("Usage: python script.py <directory> <input_nifti_file> <output_nifti_file>")
        sys.exit(1)

    directory = sys.argv[1]
    input_nifti_file = sys.argv[2]
    output_nifti_file = sys.argv[3]
    if len(sys.argv) == 5:
        threshold = float(sys.argv[4])

    try:
        if len(sys.argv) == 4:
            main(directory, input_nifti_file, output_nifti_file)
        else:
            main(directory, input_nifti_file, output_nifti_file, threshold)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
