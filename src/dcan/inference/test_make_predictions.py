import unittest
import pandas as pd

from dcan.inference.make_predictions import compute_standardized_rmse

class TestMakePredictions(unittest.TestCase):
    def test_compute_standardized_rmse(self):
        csv_data_file = "/users/9/reine097/projects/loes-scoring-2/data/anon_train_scans_and_loes.csv"
        input_df = pd.read_csv(csv_data_file)
        model_save_location = "/home/feczk001/shared/data/AlexNet/LoesScoring/loes_scoring_12.pt" 
        base_dir = "/home/feczk001/shared/projects/S1067_Loes/data/Fairview-ag/05-training_ready/" 
        subjects = ['subject-05', 'subject-50', 'subject-35', 'subject-40', 'subject-45', 'subject-62', 'subject-35', 'subject-50', 'subject-55', 'subject-62', 'subject-00', 'subject-30', 'subject-50', 'subject-35', 'subject-05', 'subject-00', 'subject-05', 'subject-55', 'subject-05', 'subject-25', 'subject-55', 'subject-30', 'subject-50', 'subject-55', 'subject-35', 'subject-55', 'subject-50', 'subject-62', 'subject-67', 'subject-15', 'subject-20', 'subject-25', 'subject-50', 'subject-15', 'subject-45', 'subject-45', 'subject-00', 'subject-55', 'subject-62', 'subject-62', 'subject-55', 'subject-67', 'subject-05', 'subject-45', 'subject-00', 'subject-55', 'subject-10', 'subject-45', 'subject-45', 'subject-45', 'subject-35', 'subject-40', 'subject-05', 'subject-25', 'subject-50', 'subject-30', 'subject-62', 'subject-50', 'subject-55', 'subject-50', 'subject-05', 'subject-35', 'subject-35', 'subject-10', 'subject-15', 'subject-20', 'subject-45', 'subject-45', 'subject-50', 'subject-45', 'subject-45', 'subject-62', 'subject-67', 'subject-45', 'subject-50', 'subject-30', 'subject-40', 'subject-62', 'subject-00', 'subject-05', 'subject-50', 'subject-67', 'subject-05', 'subject-10', 'subject-55', 'subject-45', 'subject-50', 'subject-10', 'subject-40', 'subject-15', 'subject-05', 'subject-45', 'subject-35', 'subject-55', 'subject-15']
        sessions = ['session-03', 'session-02', 'session-05', 'session-01', 'session-04', 'session-02', 'session-05', 'session-06', 'session-01', 'session-01', 'session-01', 'session-01', 'session-00', 'session-02', 'session-03', 'session-00', 'session-06', 'session-06', 'session-02', 'session-00', 'session-00', 'session-00', 'session-05', 'session-06', 'session-04', 'session-05', 'session-02', 'session-03', 'session-01', 'session-02', 'session-00', 'session-01', 'session-00', 'session-00', 'session-03', 'session-07', 'session-02', 'session-02', 'session-02', 'session-00', 'session-04', 'session-00', 'session-04', 'session-01', 'session-03', 'session-04', 'session-00', 'session-05', 'session-00', 'session-00', 'session-04', 'session-00', 'session-02', 'session-01', 'session-03', 'session-01', 'session-01', 'session-04', 'session-03', 'session-01', 'session-05', 'session-00', 'session-03', 'session-06', 'session-00', 'session-00', 'session-02', 'session-03', 'session-05', 'session-04', 'session-02', 'session-03', 'session-01', 'session-05', 'session-04', 'session-00', 'session-01', 'session-00', 'session-03', 'session-00', 'session-03', 'session-00', 'session-07', 'session-00', 'session-03', 'session-06', 'session-06', 'session-06', 'session-00', 'session-01', 'session-00', 'session-07', 'session-01', 'session-00', 'session-02']
        standardize_rmse = compute_standardized_rmse(input_df, model_save_location, base_dir, subjects, sessions)
        print(standardize_rmse)
        self.assertIsNotNone(standardize_rmse)

if __name__ == '__main__':
    unittest.main()
