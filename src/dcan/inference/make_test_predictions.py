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

from reprex.models import AlexNet3D

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)


def process_data(model_save_location, val_csv_location, val_csv_save_location):
    model = AlexNet3D(4608)
    model.load_state_dict(torch.load(model_save_location,
                                     map_location='cpu'))
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
                if math.isnan(actual_loes_score):
                    continue
            except ValueError:
                log.error(f"Loes score error on line {index + 2}")

                continue
            image = tio.ScalarImage(image_path)

            image_tensor = image.data
            output = model(image_tensor)
            prediction = output[0].item()
            df.at[index, 'prediction'] = prediction
            difference = actual_loes_score - prediction
            square = difference * difference
            squares_list.append(square)
            prediction_list.append(prediction)
        log.info(ratings_dict)
        rmse = sqrt(sum(squares_list) / len(squares_list))
        sigma = statistics.stdev(prediction_list)
        standardized_rmse = rmse / sigma
        log.info(f'standardized_rmse: {standardized_rmse}')
        df.to_csv(val_csv_save_location, index=False)
        ax1 = df.plot.scatter(x='loes_score', y = 'prediction', c = 'DarkBlue')
        plt.show()


if __name__ == "__main__":
    csv_in_folder = sys.argv[1]
    models_folder = sys.argv[2]
    csv_out_folder = sys.argv[3]
    for i in range(5):
        csv_in_file = os.path.join(csv_in_folder, f'fold{i}.csv')
        model_file = \
            os.path.join(
                models_folder, "/home/feczk001/shared/data/AlexNet/LoesScoring/loes_scoring_model_1_cv", f'fold{i}.pt')
        csv_out_file = os.path.join(csv_out_folder, f'fold{i}.csv')
        process_data(model_file, csv_in_file, csv_out_file)
