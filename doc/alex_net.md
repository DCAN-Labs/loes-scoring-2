# AlexNet 
## Architecture

This is based on Section 2.1.2 of 
*Deep Learning with PyTorch* by
Luca Pietro Giovanni Antiga, Eli Stevens, and Thomas Viehmann and
published by Manning Publications.

[TODO Add formal attributions] 

> In figure 2.3, input images come in from the left and go through five stacks of filters, each producing a number of output images. After each filter, the images are reduced in size, as annotated. The images produced by the last stack of filters are laid out as a 4,096-element 1D vector and classified to produce 1,000 output probabilities, one for each output class.

![Figure 2.3.  The AlexNet architecture](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781617295263/files/Images/CH02_F03_Stevens2_GS.png)

Mapping from AlexNet for ImageNet to our AlexNet for Loes scoring

    from reprex.models import AlexNet3D
    alexnet = AlexNet3D(4096)

    alexnet
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
        (1): Linear(in_features=4096, out_features=64, bias=True)
        (2): ReLU(inplace=True)
        (3): Dropout(p=0.5, inplace=False)
        (4): Linear(in_features=64, out_features=1, bias=True)



