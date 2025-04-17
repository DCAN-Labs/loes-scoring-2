import logging
import math
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
        log.debug(f"Input shape: {x.shape}")
        
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
        
        log.debug(f"After initial processing: {x.shape}")
        
        # Residual blocks
        x = self.layer1(x)
        log.debug(f"After layer1: {x.shape}")
        x = self.layer2(x)
        log.debug(f"After layer2: {x.shape}")
        x = self.layer3(x)
        log.debug(f"After layer3: {x.shape}")
        x = self.layer4(x)
        log.debug(f"After layer4: {x.shape}")
        
        # Global pooling and classification
        x = self.avgpool(x)
        log.debug(f"After pooling: {x.shape}")
        x = torch.flatten(x, 1)
        log.debug(f"Flattened: {x.shape}")
        x = self.fc(x)
        
        return torch.sigmoid(x)
    
class MBConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio=0.25):
        super(MBConvBlock3D, self).__init__()
        self.stride = stride
        self.use_residual = in_channels == out_channels and stride == 1
        
        # Expansion phase
        hidden_dim = in_channels * expand_ratio
        self.expand = nn.Identity() if expand_ratio == 1 else nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm3d(hidden_dim),
            nn.SiLU()
        )
        
        # Depthwise convolution
        self.depthwise = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm3d(hidden_dim),
            nn.SiLU()
        )
        
        # Squeeze and Excitation
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(hidden_dim, int(hidden_dim * se_ratio), 1),
            nn.SiLU(),
            nn.Conv3d(int(hidden_dim * se_ratio), hidden_dim, 1),
            nn.Sigmoid()
        ) if se_ratio > 0 else nn.Identity()
        
        # Output projection
        self.project = nn.Sequential(
            nn.Conv3d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels)
        )

    def forward(self, x):
        residual = x
        
        # Expansion
        x = self.expand(x)
        
        # Depthwise convolution
        x = self.depthwise(x)
        
        # Squeeze and Excitation
        x = x * self.se(x)
        
        # Projection
        x = self.project(x)
        
        # Residual connection
        if self.use_residual:
            x += residual
            
        return x

class EfficientNet3D(nn.Module):
    def __init__(self, width_multiplier=1.0, depth_multiplier=1.0, dropout_rate=0.2, debug=False):
        super(EfficientNet3D, self).__init__()
        self.debug = debug
        
        # Base network configuration
        settings = [
            # t, c, n, s, k
            [1, 16, 1, 1, 3],  # MBConv1_3x3, stride=1
            [6, 24, 2, 2, 3],  # MBConv6_3x3, stride=2
            [6, 40, 2, 2, 5],  # MBConv6_5x5, stride=2
            [6, 80, 3, 2, 3],  # MBConv6_3x3, stride=2
            [6, 112, 3, 1, 5], # MBConv6_5x5, stride=1
            [6, 192, 4, 2, 5], # MBConv6_5x5, stride=2
            [6, 320, 1, 1, 3]  # MBConv6_3x3, stride=1
        ]
        
        # Adjust channels based on width multiplier
        input_channels = self._round_filters(32, width_multiplier)
        
        # Initial stem convolution
        self.stem = nn.Sequential(
            nn.Conv3d(1, input_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(input_channels),
            nn.SiLU()
        )
        
        # Building MBConv blocks
        self.blocks = nn.Sequential()
        for t, c, n, s, k in settings:
            output_channels = self._round_filters(c, width_multiplier)
            repeats = self._round_repeats(n, depth_multiplier)
            
            # First block with stride
            self.blocks.append(MBConvBlock3D(
                input_channels, output_channels, k, s, expand_ratio=t
            ))
            
            # Remaining blocks
            for _ in range(repeats - 1):
                self.blocks.append(MBConvBlock3D(
                    output_channels, output_channels, k, 1, expand_ratio=t
                ))
                
            input_channels = output_channels
            
        # Head
        self.head = nn.Sequential(
            nn.Conv3d(input_channels, 1280, 1, bias=False),
            nn.BatchNorm3d(1280),
            nn.SiLU()
        )
        
        # Pooling and final FC
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(1280, 1)
        
    def _round_filters(self, filters, width_multiplier):
        filters *= width_multiplier
        return int(filters + 4) // 8 * 8
    
    def _round_repeats(self, repeats, depth_multiplier):
        return int(math.ceil(depth_multiplier * repeats))
    
    def forward(self, x):
        if self.debug:
            print(f"Input shape: {x.shape}")
            
        # Ensure input has 5 dimensions [batch_size, channels, D, H, W]
        if len(x.shape) == 4:
            x = x.unsqueeze(1)
            if self.debug:
                print(f"Added channel dimension: {x.shape}")
                
        # Stem
        x = self.stem(x)
        if self.debug:
            print(f"After stem: {x.shape}")
            
        # Blocks
        x = self.blocks(x)
        if self.debug:
            print(f"After blocks: {x.shape}")
            
        # Head
        x = self.head(x)
        if self.debug:
            print(f"After head: {x.shape}")
            
        # Final classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        
        return torch.sigmoid(x)


def get_advanced_mri_model(model_type='resnet3d', debug=False):
    """
    Factory function to get advanced MRI classification models
    
    Args:
        model_type (str): Type of model - 'resnet3d', 'dense3d', 'efficientnet3d', 'vit3d', 'unet3d', 'attentioncnn3d'
        debug (bool): Whether to print debug information during forward pass
        
    Returns:
        nn.Module: The requested model instance
    """
    if model_type.lower() == 'resnet3d':
        return ResNet3D(debug=debug)
    elif model_type.lower() == 'dense3d':
        return Dense3DCNN(debug=debug)
    elif model_type.lower() == 'efficientnet3d':
        return EfficientNet3D(debug=debug)
    # elif model_type.lower() == 'vit3d':
    #     return ViT3D(debug=debug)
    # elif model_type.lower() == 'unet3d':
    #     return UNet3DClassifier(debug=debug)
    # elif model_type.lower() == 'attentioncnn3d':
    #     return AttentionCNN3D(debug=debug)
    else:
        raise ValueError(f"Unknown model type: {model_type}")   