import logging
import math
import os.path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import pandas as pd
import statistics
import sys
import torch
import torchio as tio
from math import sqrt

from dcan.data_sets.dsets import LoesScoreDataset
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
        actual_scores = []
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
                actual_scores.append(actual_loes_score)
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
            ratings_dict[(subject, session,)] = prediction
            difference = actual_loes_score - prediction
            square = difference * difference
            squares_list.append(square)
            prediction_list.append(prediction)
        print(ratings_dict)
        rmse = math.sqrt(sum(squares_list) / len(squares_list))
        sigma = statistics.stdev(actual_scores)
        standardized_rmse = rmse / sigma
        log.info(f'standardized_rmse: {standardized_rmse}')
        
        return standardized_rmse
    

def predict(row):
    subject = row['anonymized_subject_id']
    session = row['anonymized_session_id']
    mprage_path = f'/home/feczk001/shared/projects/S1067_Loes/data/Fairview-ag/05-training_ready/{subject}_{session}_space-MNI_brain_mprage_RAVEL.nii.gz'
    mprage_image = tio.ScalarImage(mprage_path)
    transform = tio.Compose([
        tio.ToCanonical(),
        tio.ZNormalization(masking_method=tio.ZNormalization.mean),
    ])
    transformed_mprage_image = transform(mprage_image)
    mprage_image_tensor = transformed_mprage_image.data
    value = mprage_image_tensor.unsqueeze(0)

    return value



def process_data(model_save_location, val_csv_location):
    model = AlexNet3D(4608)
    model.load_state_dict(torch.load(model_save_location, weights_only=True,
                                     map_location='cpu'))
    model.eval()

    df = pd.read_csv(val_csv_location)
    df['prediction'] = ''
    df = df.reset_index()
    validation_rows = df.loc[df['validation'] == 1]
    output_df = validation_rows.copy()
    predictions = []
    actual_scores = []
    with torch.no_grad():
        inputs = list(output_df.apply(predict, axis=1))

        predictions = [model(input) for input in inputs]
        predict_vals = [p[0].item() for p in predictions]
        rmse = statistics.mean([(actual_scores[i] - predict_vals[i]) ** 2 for i in range(len(actual_scores))])
        sigma = statistics.stdev(actual_scores)
        standardized_rmse = rmse / sigma
        log.info(f'standardized_rmse: {standardized_rmse}')



if __name__ == "__main__":
    process_data("/home/feczk001/shared/data/AlexNet/LoesScoring/loes_scoring_12.pt", 
                 "/users/9/reine097/projects/loes-scoring-2/data/anon_train_scans_and_loes_training_test_non_gd.csv")
