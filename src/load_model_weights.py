#!/usr/bin/python

import os
import torch
import numpy as np

# traverse root directory, and list directories as dirs and files as files
for root, dirs, files in os.walk("/home/feczk001/shared/data/AlexNet/LoesScoring/"):
    path = root.split(os.sep)
    print((len(path) - 1) * '---', os.path.basename(root))
    for file in files:
        print(len(path) * '---', file)
        model_save_location = os.path.join(root, file)
        if model_save_location.endswith('.pt'):
            try:
                torch.load(model_save_location,
                                            map_location='cpu')
                print(f'Loaded: {model_save_location}')
            except Exception as inst:
                print(f'Failed: {model_save_location}')
