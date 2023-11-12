# AlexNet 
## Architecture

We are using the [DCAN-Labs / AlexNet_Abrol2021](https://github.com/DCAN-Labs/AlexNet_Abrol2021) GitHub repository for our implementation of AlexNet.

AlexNet is an instance of a [convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network) [CNN].

### Input and output layers
Inputs to AlexNet are three-dimensional MRI images.  The output, for a given MRI, is the predicted Loes score.
Loes scores range from a low of 0 to a high of 34 at increments of 0.5.

### Intermediate layers
The intermediate, hidden layers are blocks.  Each block is constructed like this (I've elided the
parameters for now):        

        (0): Conv3d(...)
        (1): BatchNorm3d(...)
        (2): ReLU(...)
        (3): MaxPool3d()

This block is a function that computes a set of output images from an input image.  The function consists of 
additions and 
multiplications combined with a non-linear function computed by `ReLU`.  

1. Conv3d(...)  # TODO Give plain English description
2. BatchNorm3d(...)  # TODO Give plain English description
3. ReLU(...)  # TODO Give plain English description
4. MaxPool3d(...)  # TODO Give plain English description

With CNNs, by constructing 
different blocks consisting of different types of constituent parts with different parameters
in different orders,  we can construct functions that are arbitrarily close to a given function.  The 
given function that we are trying to approximate in our case is the function `mri_to_loes_score` with the domain of
MRIs and the range of Loes scores.

## Formal definition of architecture

    from reprex.models import AlexNet3D
    alexnet = AlexNet3D(4608)
    alexnet
    Out[7]: 
    AlexNet3D(
      (features): Sequential(
        (0): Conv3d(1, 64, kernel_size=(5, 5, 5), stride=(2, 2, 2))
        (1): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): MaxPool3d(kernel_size=3, stride=3, padding=0, dilation=1, ceil_mode=False)
        (4): Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1))
        (5): BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (6): ReLU(inplace=True)
        (7): MaxPool3d(kernel_size=3, stride=3, padding=0, dilation=1, ceil_mode=False)
        (8): Conv3d(128, 192, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (9): BatchNorm3d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (10): ReLU(inplace=True)
        (11): Conv3d(192, 192, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (12): BatchNorm3d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (13): ReLU(inplace=True)
        (14): Conv3d(192, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (15): BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (16): ReLU(inplace=True)
        (17): MaxPool3d(kernel_size=3, stride=3, padding=0, dilation=1, ceil_mode=False)
      )
      (classifier): Sequential(
        (0): Dropout(p=0.5, inplace=False)
        (1): Linear(in_features=4608, out_features=64, bias=True)
        (2): ReLU(inplace=True)
        (3): Dropout(p=0.5, inplace=False)
        (4): Linear(in_features=64, out_features=1, bias=True)
      )
    )