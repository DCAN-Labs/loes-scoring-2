import logging
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)


class MRILogisticRegressionModel(nn.Module):
    """
    A logistic regression model modified to work with raw MRI data.
    This model includes a dimensionality reduction step before applying logistic regression.
    """
    def __init__(self, debug=False):
        super(MRILogisticRegressionModel, self).__init__()
        self.debug = debug
        
        # Define a dimensionality reduction pathway
        # First, apply convolutional layers to reduce spatial dimensions
        self.conv_layers = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((4, 4, 4))  # Reduce to fixed size regardless of input dimensions
        )
        
        # Calculate the size of the flattened feature vector
        # For adaptive pooling to (4,4,4) with 32 channels: 32*4*4*4 = 2048
        self.flattened_size = 32 * 4 * 4 * 4
        
        # Apply the logistic regression on the reduced feature space
        self.classifier = nn.Linear(self.flattened_size, 1)
        
    def forward(self, x):
        # Input shape debugging
        log.debug(f"Input shape: {x.shape}")
        
        # Ensure input has 5 dimensions: [batch_size, channels, depth, height, width]
        if len(x.shape) == 4:
            # If input is [batch_size, depth, height, width], add channel dimension
            x = x.unsqueeze(1)
        
        log.debug(f"Adjusted input shape: {x.shape}")
        
        # Apply convolutional feature extraction
        x = self.conv_layers(x)
        
        log.debug(f"After conv layers: {x.shape}")
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        
        log.debug(f"After flattening: {x.shape}")
        
        # Apply logistic regression
        logits = self.classifier(x)
        
        log.debug(f"Logits shape: {logits.shape}")
        
        # Apply sigmoid to get probabilities
        return torch.sigmoid(logits)


class SimpleMRILogisticRegression(nn.Module):
    """
    A simpler approach that directly flattens the MRI and applies dimensionality reduction
    through a sequence of linear layers before the final logistic regression.
    
    Note: This approach may be less effective than using convolutional layers for MRI data
    but maintains the spirit of logistic regression.
    """
    def __init__(self, reduction_factor=10, debug=False):
        super(SimpleMRILogisticRegression, self).__init__()
        self.debug = debug
        
        # Add a fixed spatial reduction first (like adaptive pooling)
        self.spatial_reducer = nn.AdaptiveAvgPool3d((16, 16, 16))
        # Fixed input size after spatial reduction and flattening: 16*16*16 = 4096
        
        # Now we can pre-initialize with known size
        self.feature_reducer = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU()
        )
        self.classifier = nn.Linear(64, 1)
        self.initialized = False
        self.reduction_factor = reduction_factor


    def _initialize(self, input_size):
        # Create a dimensionality reduction layer
        reduced_size = max(1, input_size // self.reduction_factor)
        
        # Create a sequence of reduction layers
        self.feature_reducer = nn.Sequential(
            nn.Linear(input_size, input_size // 4),
            nn.ReLU(),
            nn.Linear(input_size // 4, input_size // 16),
            nn.ReLU(),
            nn.Linear(input_size // 16, reduced_size),
            nn.ReLU()
        )
        
        # Final classifier layer
        self.classifier = nn.Linear(reduced_size, 1)
        
        self.initialized = True
        
    def forward(self, x):
        # Log the input shape
        log.debug(f"Input shape: {x.shape}")
        
        # Flatten the input
        x_flat = x.view(x.size(0), -1)
        
        if self.debug:
            print(f"Flattened shape: {x_flat.shape}")
        
        # Initialize the network if it's the first forward pass
        if not self.initialized:
            self._initialize(x_flat.size(1))
        
        # Apply dimensionality reduction
        x_reduced = self.feature_reducer(x_flat)
        
        if self.debug:
            print(f"Reduced shape: {x_reduced.shape}")
        
        # Apply logistic regression
        logits = self.classifier(x_reduced)
        
        log.debug(f"Logits shape: {logits.shape}")
        
        # Apply sigmoid to get probabilities
        return torch.sigmoid(logits)


# You can choose which model to use based on your needs
def get_mri_logistic_regression_model(model_type='conv', debug=False):
    """
    Factory function to get the appropriate MRI logistic regression model.
    
    Args:
        model_type (str): Type of model to use - 'conv' for convolutional approach or 'simple' for direct flattening
        debug (bool): Whether to print debug information during forward pass
        
    Returns:
        nn.Module: The requested model instance
    """
    if model_type.lower() == 'conv':
        return MRILogisticRegressionModel(debug=debug)
    elif model_type.lower() == 'simple':
        return SimpleMRILogisticRegression(debug=debug)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'conv' or 'simple'.")