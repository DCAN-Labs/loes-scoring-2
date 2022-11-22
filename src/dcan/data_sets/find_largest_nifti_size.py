import glob
import torchio as tio

loes_score_images_path = '/home/feczk001/shared/data/loes_scoring/Loes_score/sub-*/ses-*/'
nifti_ext = '.nii.gz'
mprage_mri_list = glob.glob(f'{loes_score_images_path}mprage{nifti_ext}')
max_sizes = [0] * 4
for image_path in mprage_mri_list:
    image = tio.ScalarImage(image_path)
    shape = image.shape
    for i in range(4):
        if shape[i] > max_sizes[i]:
            max_sizes[i] = shape[i]
print(max_sizes)
