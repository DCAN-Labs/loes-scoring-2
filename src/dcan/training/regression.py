import logging
import os
import sys
import torch
import pandas as pd
import numpy as np
from monai.config import print_config
from monai.data import DataLoader, ImageDataset
from monai.networks.nets import Regressor
from monai.transforms import Compose, ScaleIntensity, EnsureChannelFirst, Resize
from torch.utils.tensorboard import SummaryWriter

from dcan.metrics import get_standardized_rmse

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Check CUDA availability
pin_memory = torch.cuda.is_available()
device = torch.device("cuda" if pin_memory else "cpu")

# Print MONAI configuration
print_config()

# Define constants for image size
IMAGE_SIZE = (197, 233, 189)

def load_data(loes_scoring_folder):
    """Loads and processes the data from the provided folder."""
    df = pd.read_csv(os.path.join(loes_scoring_folder, 'Nascene_deID_files.csv'))
    defaced_folder = os.path.join(loes_scoring_folder, 'nascene_deid/BIDS/defaced/')

    # Add the full path for each image
    df['full_path'] = df['FILE'].apply(lambda x: os.path.join(defaced_folder, x))
    
    # Prepare image file paths and labels
    images = df['full_path'].tolist()
    loes_scores = np.array([[score] for score in df['loes_score']])

    return images, loes_scores

def create_data_loaders(images, loes_scores):
    """Creates data loaders for training and validation."""
    # Define transforms
    train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize(IMAGE_SIZE)])
    val_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize(IMAGE_SIZE)])

    # Create datasets and loaders
    train_ds = ImageDataset(image_files=images[:10], labels=loes_scores[:10], transform=train_transforms)
    val_ds = ImageDataset(image_files=images[-10:], labels=loes_scores[-10:], transform=val_transforms)
    
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=2, num_workers=2, pin_memory=pin_memory)

    return train_loader, val_loader

def create_model():
    """Creates and returns the regression model."""
    model = Regressor(in_shape=[1] + list(IMAGE_SIZE), out_shape=[1], 
                      channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2))
    if torch.cuda.is_available():
        model.cuda()
    return model

def train_one_epoch(model, train_loader, loss_function, optimizer, writer, epoch):
    """Trains the model for one epoch and logs training loss."""
    model.train()
    epoch_loss = 0
    for step, batch_data in enumerate(train_loader, start=1):
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels.float())
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        writer.add_scalar("train_loss", loss.item(), epoch * len(train_loader) + step)

    return epoch_loss / len(train_loader)

def validate(model, val_loader, loss_function):
    """Validates the model and calculates RMSE."""
    model.eval()
    all_labels = []
    all_val_outputs = []

    with torch.no_grad():
        for val_data in val_loader:
            val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
            all_labels.extend(val_labels.cpu().detach().numpy())
            val_outputs = model(val_images)
            all_val_outputs.extend(val_outputs.cpu().detach().numpy().flatten())

    mse = np.mean(np.square(np.subtract(all_labels, all_val_outputs)))
    rmse = np.sqrt(mse)
    
    return rmse, all_labels, all_val_outputs

def save_best_model(model, rmse, lowest_rmse, epoch):
    """Saves the model if it achieves the best RMSE."""
    if rmse < lowest_rmse:
        lowest_rmse = rmse
        torch.save(model.state_dict(), "best_metric_model.pth")
        log.info(f"Saved new best model with RMSE: {rmse:.4f} at epoch {epoch + 1}")
    return lowest_rmse

def main(loes_scoring_folder):
    # Load data
    images, loes_scores = load_data(loes_scoring_folder)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(images, loes_scores)
    
    # Create model and optimizer
    model = create_model()
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # TensorBoard writer
    writer = SummaryWriter()

    # Training loop
    max_epochs = 5
    lowest_rmse = float('inf')
    
    for epoch in range(max_epochs):
        log.info(f"Epoch {epoch + 1}/{max_epochs}")
        
        # Train for one epoch
        epoch_loss = train_one_epoch(model, train_loader, loss_function, optimizer, writer, epoch)
        log.info(f"Epoch {epoch + 1} - Average train loss: {epoch_loss:.4f}")

        # Validate the model
        rmse, all_labels, all_val_outputs = validate(model, val_loader, loss_function)
        log.info(f"Epoch {epoch + 1} - Validation RMSE: {rmse:.4f}")

        # Save the best model based on RMSE
        lowest_rmse = save_best_model(model, rmse, lowest_rmse, epoch)

        # Optionally compute and log standardized RMSE
        try:
            standardized_rmse = get_standardized_rmse(all_labels, all_val_outputs)
            log.info(f"Standardized RMSE: {standardized_rmse}")
        except ZeroDivisionError as err:
            log.error(f"Could not compute standardized RMSE because sigma is 0: {err}")

        # Log RMSE to TensorBoard
        writer.add_scalar("val_rmse", rmse, epoch + 1)
    
    log.info(f"Training completed - Lowest RMSE: {lowest_rmse:.4f}")
    writer.close()

if __name__ == '__main__':
    loes_scoring_folder = sys.argv[1]
    main(loes_scoring_folder)
