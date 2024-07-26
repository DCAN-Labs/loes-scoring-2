# https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html

from functools import partial
import os
import tempfile
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle
from monai.transforms import (
    EnsureChannelFirst,
    Resize,
    ScaleIntensity,
)
from monai.data import DataLoader, ImageDataset
import pandas as pd
import numpy as np
from monai.networks.nets import Regressor
import random
def add_folder(row, defaced_folder):
    return os.path.join(defaced_folder, row['FILE'])

def load_data(images):
    transform = transforms.Compose(
        [ScaleIntensity(), EnsureChannelFirst(), Resize((197, 233, 189))])

    n = len(images)
    k = int(round(0.8 * n))i
    random_indices = random.sample(range(n), k)
    trainset = \
        ImageDataset(image_files=[images[i] for i in random_indices], 
                     labels=[loes_scores[i] for i in random_indices], 
                     transform=transform)
    testset = \
        ImageDataset(image_files=[images[i] for i in range(n) if i not in random_indices], 
                     labels=[loes_scores[i] for i in range(n) if i not in random_indices], 
                     transform=transform)

    return trainset, testset

class Net(Regressor):
    def __init__(self, in_shape, out_shape, channels, strides, num_res_units: int = 2):
        super(Net, self).__init__(in_shape, out_shape, channels, strides, num_res_units=num_res_units)

def measure_rmse(net, device="cpu"):
    _, testset = load_data()

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2
    )

    all_labels = []
    all_val_outputs = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            all_labels.extend(labels.cpu().detach().numpy())
            flattened_val_outputs = [val for sublist in outputs.cpu().detach().numpy() for val in sublist]
            all_val_outputs.extend(flattened_val_outputs)

    mse = np.square(np.subtract(all_labels, all_val_outputs)).mean()
    rmse = np.sqrt(mse)

    return rmse


def train_loes_scoring(config):
    net = Net(in_shape=[1, 197, 233, 189], out_shape=[1], channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2))

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

   # It is important that we use nn.MSELoss for regression.
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]
            net.load_state_dict(checkpoint_state["net_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    trainset, testset = load_data()

    test_abs = int(len(testset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs]
    )

    trainloader = torch.utils.data.DataLoader(
        train_subset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=8
    )
    valloader = torch.utils.data.DataLoader(
        val_subset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=8
    )

    for epoch in range(start_epoch, 10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs.float(), labels.float())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, i + 1, running_loss / epoch_steps)
                )
                running_loss = 0.0

        # Validation loss
        all_labels = []
        all_val_outputs = []
        val_loss = 0.0
        val_steps = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                flattened_val_outputs = [np.double(val) for sublist in outputs.cpu().detach().numpy() for val in sublist]
                all_val_outputs.extend(flattened_val_outputs)
                flattened_label_outputs = [np.double(val) for sublist in labels.cpu().detach().numpy() for val in sublist]
                all_labels.extend(flattened_label_outputs)

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        mse = np.square(np.subtract(all_labels, all_val_outputs)).mean()
        rmse = np.sqrt(mse)

        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "wb") as fp:
                pickle.dump(checkpoint_data, fp)

            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            train.report(
                {"loss": val_loss / val_steps, "rmse": rmse},
                checkpoint=checkpoint,
            )

    print("Finished Training")
    
def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    loes_scoring_folder = '/home/feczk001/shared/data/loes_scoring/'
    df = pd.read_csv(os.path.join(loes_scoring_folder, 'Nascene_deID_files.csv'))

    defaced_folder = os.path.join(loes_scoring_folder, 'nascene_deid/BIDS/defaced/')

    df['full_path'] = df.apply(add_folder, axis=1)

    # Setup data directory
    root_dir = '/home/feczk001/shared/data/loes_scoring/nascene_deid/BIDS/defaced/'
    images = list(df['full_path'])

    loes_scores = np.array(
        [[score] for score in df['loes_score']]
    )

    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]
            net.load_state_dict(checkpoint_state["net_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    load_data()

    config = {
        "lr": tune.loguniform(1e-4, 1e-3),
        "batch_size": tune.choice([1, 2, 4]),
        "weight_decay": tune.loguniform(1e-4, 1e-3),
        "num_res_units": tune.choice([1, 2, 3])
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )
    result = tune.run(
        partial(train_loes_scoring),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation rmse: {best_trial.last_result['rmse']}")

    best_trained_model = Net(in_shape=[1, 197, 233, 189], out_shape=[1], channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2))
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint = result.get_best_checkpoint(trial=best_trial, metric="rmse", mode="max")
    with best_checkpoint.as_directory() as checkpoint_dir:
        data_path = Path(checkpoint_dir) / "data.pkl"
        with open(data_path, "rb") as fp:
            best_checkpoint_data = pickle.load(fp)

        best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])
        test_rmse = measure_rmse(best_trained_model, device)
        print("Best trial test set rmse: {}".format(test_rmse))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=10, max_num_epochs=10, gpus_per_trial=0)
