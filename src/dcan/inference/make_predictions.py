import logging
import math
import os.path

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
    df['prediction'] = ''
    df = df.reset_index()
    ratings_dict = dict()
    with torch.no_grad():
        squares_list = []
        prediction_list = []
        for index, row in df.iterrows():
            if row['QC'] == '1':
                image_path = row['FILE']
                try:
                    actual_loes_score = row['loes_score']
                    if math.isnan(actual_loes_score):
                        continue
                except ValueError:
                    log.error(f"Loes score error on line {index + 2}")

                    continue
                image_path_parts = image_path.split('/')
                image_file = image_path_parts[-1]
                image_path = \
                    os.path.join('/home/feczk001/shared/data/loes_scoring/nascene_deid/BIDS/defaced/', image_file)
                image = tio.ScalarImage(image_path)

                image_tensor = image.data

                image_tensor = torch.unsqueeze(image_tensor, dim=0)
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


if __name__ == "__main__":
    process_data(sys.argv[1], sys.argv[2], sys.argv[3])
