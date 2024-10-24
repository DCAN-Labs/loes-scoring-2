from torchvision.models import (
        resnet18, ResNet18_Weights
)
from PIL import Image
import torch


weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
transforms = weights.transforms()

image = Image.open("img/MollySmall.jpg")
image_tensor = transforms(image)
image_reshaped = image_tensor.unsqueeze(0)

model.eval()
with torch.no_grad():
        pred = model(image_reshaped).squeeze(0)
        pred_cls = pred.softmax(0)
        cls_id = pred_cls.argmax().item()
        cls_name = weights.meta["categories"][cls_id]
        print(cls_name)
        