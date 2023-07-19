import numpy as np
import torch.utils.data
import torchio as tio
from captum.attr import Saliency

from dcan.data_sets.dsets import LoesScoreDataset
from reprex.models import AlexNet3D


def normalize_array(array):
    new_array = (array - array.min()) / (array.max() - array.min())

    return new_array


net = AlexNet3D(4608)
model_save_location = '/home/miran045/reine097/projects/loes-scoring-explainability/models/loes_scoring_03.pt'
net.load_state_dict(torch.load(model_save_location,
                               map_location='cpu'))
net.eval()

example_file = \
    '/home/feczk001/shared/projects/S1067_Loes/data/MNI-space_Loes_data/' \
    'sub-4750MASZ_ses-20080220_space-MNI_mprageGd.nii.gz'
image = tio.ScalarImage(example_file)

image_tensor = image.data

image_tensor = torch.unsqueeze(image_tensor, dim=0)
image_tensor = normalize_array(image_tensor)

with torch.no_grad():
    output = net(image_tensor)
    print(output)

print("Using existing trained model")
net.load_state_dict(torch.load('models/loes_scoring_03.pt'))

csv_data_file = "./data/MNI-space_Loes_data.csv"
testset = LoesScoreDataset(
    csv_data_file,
    use_gd_only=False,
    val_stride=10,
    is_val_set_bool=True,
)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

dataiter = iter(testloader)
images, labels = next(dataiter)

ind = 3

input = images[ind].unsqueeze(0)
input.requires_grad = True

net.eval()

def attribute_image_features(algorithm, input, **kwargs):
    net.zero_grad()
    tensor_attributions = algorithm.attribute(input,
                                              target=labels[ind],
                                              **kwargs
                                              )

    return tensor_attributions

def wrapped_model(inp):
    return net(inp)[0]

saliency = Saliency(wrapped_model)
grads = saliency.attribute(input)
grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))
print(grads)