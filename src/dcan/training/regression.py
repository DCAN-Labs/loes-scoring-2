# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/miran045/reine097/projects/loes-scoring-2/.venv/lib/python3.11/site-packages/nvidia/cudnn/lib/
# ~/projects/loes-scoring-2/.venv/bin/python  ./dcan/training/regression.py
# Setup imports

import logging
import os
import sys

import monai
import numpy as np
import pandas as pd
import torch
from monai.config import print_config
from monai.data import DataLoader, ImageDataset
from monai.networks.nets import Regressor
from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    RandRotate90,
    Resize,
    ScaleIntensity,
)
from torch.utils.tensorboard import SummaryWriter

from dcan.metrics import get_standardized_rmse

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)

pin_memory = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
print_config()

loes_scoring_folder = '/home/feczk001/shared/data/loes_scoring/'
df = pd.read_csv(os.path.join(loes_scoring_folder, 'Nascene_deID_files.csv'))

defaced_folder = os.path.join(loes_scoring_folder, 'nascene_deid/BIDS/defaced/')


def add_folder(row):
    return os.path.join(defaced_folder, row['FILE'])


df['full_path'] = df.apply(add_folder, axis=1)

# Setup data directory
root_dir = '/home/feczk001/shared/data/loes_scoring/nascene_deid/BIDS/defaced/'
images = list(df['full_path'])

loes_scores = np.array(
    [[score] for score in df['loes_score']]
)

# Define transforms
train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((197, 233, 189))])

val_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((197, 233, 189))])

# Define nifti dataset, data loader
check_ds = ImageDataset(image_files=images, labels=loes_scores, transform=train_transforms)
check_loader = DataLoader(check_ds, batch_size=3, num_workers=2, pin_memory=pin_memory)

im, label = monai.utils.misc.first(check_loader)
print(type(im), im.shape, label, label.shape)

# create a training data loader
train_ds = ImageDataset(image_files=images[:10], labels=loes_scores[:10], transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2, pin_memory=pin_memory)

# create a validation data loader
val_ds = ImageDataset(image_files=images[-10:], labels=loes_scores[-10:], transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=2, num_workers=2, pin_memory=pin_memory)

model = Regressor(in_shape=[1, 197, 233, 189], out_shape=[1], channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2))
if torch.cuda.is_available():
    model.cuda()
# It is important that we use nn.MSELoss for regression.
loss_function = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), 1e-4)

# start a typical PyTorch training
val_interval = 2
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
writer = SummaryWriter()
max_epochs = 5

lowest_rmse = sys.float_info.max
for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0

    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels.float())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_len = len(train_ds) // train_loader.batch_size
        print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        all_labels = []
        all_val_outputs = []
        for val_data in val_loader:
            val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
            all_labels.extend(val_labels.cpu().detach().numpy())
            with torch.no_grad():
                val_outputs = model(val_images)
                flattened_val_outputs = [val for sublist in val_outputs.cpu().detach().numpy() for val in sublist]
                all_val_outputs.extend(flattened_val_outputs)

        mse = np.square(np.subtract(all_labels, all_val_outputs)).mean()
        rmse = np.sqrt(mse)

        if rmse < lowest_rmse:
            lowest_rmse = rmse
            lowest_rmse_epoch = epoch + 1
            torch.save(model.state_dict(), "best_metric_model_classification3d_array.pth")
            print("saved new best metric model")
            try:
                standardized_rmse = get_standardized_rmse(all_labels, all_val_outputs)
                log.info(f'standardized_rmse: {standardized_rmse}')
            except ZeroDivisionError as err:
                log.error(f'Could not compute standardized RMSE because sigma is 0: {err}')

        print(f"Current epoch: {epoch + 1} current RMSE: {rmse:.4f} ")
        print(f"Best RMSE: {lowest_rmse:.4f} at epoch {lowest_rmse_epoch}")
        writer.add_scalar("val_rmse", rmse, epoch + 1)

print(f"Training completed, lowest_rmse: {lowest_rmse:.4f} at epoch: {lowest_rmse_epoch}")
writer.close()
