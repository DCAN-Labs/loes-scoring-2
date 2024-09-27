import logging
import math
import os.path
import matplotlib.pyplot as plt

import pandas as pd
import statistics
import sys
import torch
import torchio as tio
from math import sqrt
import numpy as np

from dcan.inference.models import AlexNet3D

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)


def process_data(model_save_location, val_csv_location, val_csv_save_location):
    model = AlexNet3D(4608)
    weights = torch.load(model_save_location,
                                     map_location='cpu')
    model.load_state_dict(weights)
    model.eval()

    df = pd.read_csv(val_csv_location)
    ratings_dict = dict()
    with torch.no_grad():
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        squares_list = []
        prediction_list = []
        for index, row in df.iterrows():
            if row['train/validation/test'] != 'test':
                continue
            image_path = row['file']
            try:
                actual_loes_score = row['loes-score']
            except ValueError:
                log.error(f"Loes score error on line {index + 2}")

                continue
            image = tio.ScalarImage(image_path)

            # image_tensor = image.data.to(device)
            input_tensor = image.data.float() 
            image_tensor = torch.unsqueeze(input_tensor, dim=0)

            print(image_tensor.dtype)
            output = model(image_tensor)
            prediction = output[0].item()
            prediction_list.append(prediction)
            df.at[index, 'prediction'] = prediction
            if math.isnan(actual_loes_score):
                continue
            difference = actual_loes_score - prediction
            square = difference * difference
            squares_list.append(square)
        log.info(ratings_dict)
        rmse = sqrt(sum(squares_list) / len(squares_list))
        sigma = statistics.stdev(prediction_list)
        standardized_rmse = rmse / sigma
        log.info(f'standardized_rmse: {standardized_rmse}')
        df.to_csv(val_csv_save_location, index=False)
        ax1 = df.plot.scatter(x='loes_score', y = 'prediction', c = 'DarkBlue')
        plt.show()


if __name__ == "__main__":
    csv_in_file = '/users/9/reine097/loes_scoring/in.csv'
    model_file = '/home/feczk001/shared/data/AlexNet/LoesScoring/loes_scoring_09_256.pt'
    csv_out_file = '/users/9/reine097/loes_scoring/out.csv'
    process_data(model_file, csv_in_file, csv_out_file)
