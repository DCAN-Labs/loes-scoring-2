import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path


# Define image and histogram directories
img_dir = Path('/home/feczk001/shared/projects/S1067_Loes/data/niftis_deID/masked/non_gd/gm/')
img_name = 'masked-sub-01_ses-01_space-MNI_brain_normalized_mprage.nii.gz'
img_path = img_dir / img_name

# Load NIfTI image
img = nib.load(img_path)
gray_matter_data = img.get_fdata()

# Flatten image data and filter out zero intensities
flattened_data = gray_matter_data.flatten()
df = pd.DataFrame(flattened_data, columns=['voxel_intensity'])
filtered_data = df[df['voxel_intensity'] > 0]

# Plot histogram
plt.figure(figsize=(10, 6))
sns.histplot(data=filtered_data, x='voxel_intensity', bins=30, color='purple')
plt.title(f'Distribution of gray matter voxel intensities in {img_name}')
plt.xlabel('Voxel intensity')
plt.ylabel('Frequency')
plt.grid(True)

# Save the plot
hist_dir = Path('/home/feczk001/shared/projects/S1067_Loes/data/niftis_deID/histograms/non_gd/gm/')
hist_file_name = f'{img_name[:-7]}.png'  # Strip '.nii.gz' from the name
hist_path = hist_dir / hist_file_name
hist_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(hist_path)
