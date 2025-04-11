import logging
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(self._make_layer(in_channels + i * growth_rate, growth_rate))
    
    def _make_layer(self, in_channels, growth_rate):
        return nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        )
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, 1))
            features.append(new_feature)
        return torch.cat(features, 1)

class Dense3DCNN(nn.Module):
    """
    3D DenseNet architecture specialized for MRI classification
    """
    def __init__(self, growth_rate=16, block_config=(3, 3, 3), debug=False):
        super(Dense3DCNN, self).__init__()
        self.debug = debug
        
        # Initial convolution
        self.features = nn.Sequential(
            nn.Conv3d(1, growth_rate * 2, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(growth_rate * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        
        # Dense blocks
        in_channels = growth_rate * 2
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(in_channels, growth_rate, num_layers)
            self.features.add_module(f'denseblock{i+1}', block)
            in_channels += num_layers * growth_rate
            
            # Add transition layers between dense blocks (except after the last block)
            if i != len(block_config) - 1:
                self.features.add_module(f'transition{i+1}', nn.Sequential(
                    nn.BatchNorm3d(in_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(in_channels, in_channels // 2, kernel_size=1, stride=1, bias=False),
                    nn.AvgPool3d(kernel_size=2, stride=2)
                ))
                in_channels = in_channels // 2
        
        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(in_channels))
        self.features.add_module('relu5', nn.ReLU(inplace=True))
        
        # Global average pooling and classifier
        self.adaptive_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(in_channels, 1)
        
    def forward(self, x):
        if self.debug:
            print(f"Input shape: {x.shape}")
        
        # Ensure input has 5 dimensions [batch_size, channels, D, H, W]
        if len(x.shape) == 4:
            x = x.unsqueeze(1)
            if self.debug:
                print(f"Added channel dimension: {x.shape}")
        
        # Feature extraction
        features = self.features(x)
        if self.debug:
            print(f"Features shape: {features.shape}")
        
        # Global pooling
        out = self.adaptive_pool(features)
        if self.debug:
            print(f"After pooling: {out.shape}")
        
        # Flatten and classify
        out = torch.flatten(out, 1)
        if self.debug:
            print(f"Flattened: {out.shape}")
        
        out = self.classifier(out)
        if self.debug:
            print(f"Classifier output: {out.shape}")
        
        return torch.sigmoid(out)


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock3D, self).__init__()
        # First convolution (may change dimensions)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Second convolution (keeps dimensions the same)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(identity)
        out = self.relu(out)
        
        return out
    
class ResNet3D(nn.Module):
    """
    3D ResNet architecture for MRI classification
    """
    def __init__(self, block=ResidualBlock3D, num_blocks=[2, 2, 2, 2], debug=False):
        super(ResNet3D, self).__init__()
        self.debug = debug
        self.in_channels = 64
        
        # Initial convolution
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(512, 1)
        
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        if self.debug:
            print(f"Input shape: {x.shape}")
        
        # Ensure input has 5 dimensions [batch_size, channels, D, H, W]
        if len(x.shape) == 4:
            x = x.unsqueeze(1)
            if self.debug:
                print(f"Added channel dimension: {x.shape}")
        
        # Initial processing
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        if self.debug:
            print(f"After initial processing: {x.shape}")
        
        # Residual blocks
        x = self.layer1(x)
        if self.debug:
            print(f"After layer1: {x.shape}")
        x = self.layer2(x)
        if self.debug:
            print(f"After layer2: {x.shape}")
        x = self.layer3(x)
        if self.debug:
            print(f"After layer3: {x.shape}")
        x = self.layer4(x)
        if self.debug:
            print(f"After layer4: {x.shape}")
        
        # Global pooling and classification
        x = self.avgpool(x)
        if self.debug:
            print(f"After pooling: {x.shape}")
        x = torch.flatten(x, 1)
        if self.debug:
            print(f"Flattened: {x.shape}")
        x = self.fc(x)
        
        return torch.sigmoid(x)


def get_advanced_mri_model(model_type='resnet3d', debug=False):
    """
    Factory function to get advanced MRI classification models
    
    Args:
        model_type (str): Type of model - 'resnet3d', 'dense3d'
        debug (bool): Whether to print debug information during forward pass
        
    Returns:
        nn.Module: The requested model instance
    """
    if model_type.lower() == 'resnet3d':
        return ResNet3D(debug=debug)
    elif model_type.lower() == 'dense3d':
        return Dense3DCNN(debug=debug)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'resnet3d' or 'dense3d'.")
    