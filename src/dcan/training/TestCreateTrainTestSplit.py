import unittest

from dcan.training import create_train_test_split

class TestCreateTrainTestSplit(unittest.TestCase):

    def test_create_train_test_split(self):
        csv_data_file_in = "/users/9/reine097/projects/loes-scoring-2/data/anon_train_scans_and_loes.csv"
        csv_data_file_out = "/users/9/reine097/projects/loes-scoring-2/data/anon_train_scans_and_loes_training_test_non_gd.csv"
        create_train_test_split.create_train_test_split(csv_data_file_in, csv_data_file_out)


if __name__ == '__main__':
    unittest.main()
