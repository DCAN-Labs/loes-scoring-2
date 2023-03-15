# Based on https://captum.ai/

import nibabel as nib
import torch
import tensorflow as tf
from reprex.models import AlexNet3DDropoutRegression
from captum.attr import IntegratedGradients


example_file_score_21 = \
    '/home/feczk001/shared/projects/S1067_Loes/data/MNI-space_Loes_data/' \
    'sub-6630SICH_ses-20160727_space-MNI_mprageGd.nii.gz'
img = nib.load(example_file_score_21)
image_data = img.get_fdata()

input = tf.convert_to_tensor(image_data)
baseline = torch.zeros(197, 233, 189)

model_save_location = "/home/feczk001/shared/data/AlexNet/LoesScoring/loes_scoring_test.pt"

model = AlexNet3DDropoutRegression(4608)
# reload weights on non-parallel model
model.load_state_dict(torch.load(model_save_location))
model.eval()

ig = IntegratedGradients(model)
attributions, delta = ig.attribute(input, baseline, target=0, return_convergence_delta=True)
print('IG Attributions:', attributions)
print('Convergence Delta:', delta)
