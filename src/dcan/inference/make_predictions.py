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

from dcan.inference.models import AlexNet3D

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)





def compute_standardized_rmse(input_df, model_save_location, base_dir, subjects, sessions):
    model = AlexNet3D(4608)
    model.load_state_dict(torch.load(model_save_location,
                                     map_location='cpu'))
    model.eval()

    ratings_dict = dict()
    with torch.no_grad():
        squares_list = []
        prediction_list = []
        for i in range(len(subjects)):
            subject = subjects[i]
            session = sessions[i]
            image_path = f'{base_dir}/{subject}_{session}_space-MNI_brain_mprage_RAVEL.nii.gz'
            print(image_path)
            try:
                df_multiple_and = \
                    input_df[
                        (input_df['anonymized_subject_id'] == subject) & (input_df['anonymized_session_id'] == session)]
                print(f'df_multiple_and: {df_multiple_and}')
                row = df_multiple_and.iloc[0]
                actual_loes_score = row['loes-score']
                if math.isnan(actual_loes_score):
                    continue
            except ValueError:
                log.error(f"Loes score error")

                continue
            image = tio.ScalarImage(image_path)

            image_tensor = image.data

            image_tensor = torch.unsqueeze(image_tensor, dim=0)
            output = model(image_tensor)
            prediction = output[0].item()
            difference = actual_loes_score - prediction
            square = difference * difference
            squares_list.append(square)
            prediction_list.append(prediction)
        log.info(ratings_dict)
        rmse = math.sqrt(sum(squares_list) / len(squares_list))
        sigma = statistics.stdev(prediction_list)
        standardized_rmse = rmse / sigma
        log.info(f'standardized_rmse: {standardized_rmse}')
        
        return standardized_rmse



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
            image_path = row['file']
            try:
                actual_loes_score = row['loes-score']
                if math.isnan(actual_loes_score):
                    continue
            except ValueError:
                log.error(f"Loes score error on line {index + 2}")

                continue
            image_path = row['file']
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
        ax1 = df.plot.scatter(x='loes_score', y = 'prediction', c = 'DarkBlue')
        plt.show()


if __name__ == "__main__":
    process_data(sys.argv[1], sys.argv[2], sys.argv[3])
