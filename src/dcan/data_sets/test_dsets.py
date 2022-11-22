import unittest
from datetime import datetime

from dcan.loes_scoring.data_sets.dsets import get_candidate_info_list, LoesScoreMRIs, get_loes_score_mris, \
    get_mri_raw_candidate, LoesScoreDataset

from dcan.loes_scoring.data_sets.dsets import CandidateInfoTuple

scores_igor_updated_csv = \
    '/home/miran045/reine097/projects/AlexNet_Abrol2021/data/loes_scoring/' + \
    'ALD-google_sheet-Jul272022-Loes_scores-Igor_updated.csv'


class TestDSets(unittest.TestCase):

    def test_get_candidate_info_list(self):
        candidate_info_list = \
            get_candidate_info_list(
                scores_igor_updated_csv)
        self.assertIsNotNone(candidate_info_list)

    def test_LoesScoreMRIs_init(self):
        loes_score_float = 10.0
        subject_session_uid = '7295TASU_20190220'
        subject_str = '7295TASU'
        session_str = '20190220'
        session_date = datetime.strptime(session_str, '%Y%m%d')
        is_validated = False
        candidate_info = CandidateInfoTuple(
                loes_score_float,
                subject_session_uid,
                subject_str,
                session_str,
                session_date,
                is_validated
            )
        loes_score_mris = LoesScoreMRIs(candidate_info, True)
        self.assertIsNotNone(loes_score_mris)
        raw_candidate = loes_score_mris.get_raw_candidate()
        self.assertIsNotNone(raw_candidate)

    def test_LoesScoreMRIs_init_training(self):
        loes_score_float = 10.0
        subject_session_uid = '7295TASU_20190220'
        subject_str = '7295TASU'
        session_str = '20190220'
        session_date = datetime.strptime(session_str, '%Y%m%d')
        is_validated = False
        candidate_info = CandidateInfoTuple(
                loes_score_float,
                subject_session_uid,
                subject_str,
                session_str,
                session_date,
                is_validated
            )
        loes_score_mris = LoesScoreMRIs(candidate_info, False)
        self.assertIsNotNone(loes_score_mris)
        raw_candidate = loes_score_mris.get_raw_candidate()
        self.assertIsNotNone(raw_candidate)

    def test_get_loes_score_mris(self):
        candidate_info = self.get_test_candidate_info()
        loes_score_mris = get_loes_score_mris(candidate_info, True)
        self.assertIsNotNone(loes_score_mris)

    def test_get_loes_score_mris_training(self):
        candidate_info = self.get_test_candidate_info()
        loes_score_mris = get_loes_score_mris(candidate_info, False)
        self.assertIsNotNone(loes_score_mris)

    @staticmethod
    def get_test_candidate_info():
        loes_score_float = 2.0
        subject_session_uid = '5772LAVA_20180828'
        subject_str = '5772LAVA'
        session_str = '20180828'
        session_date = datetime.strptime(session_str, '%Y%m%d')
        is_validated = False
        candidate_info = CandidateInfoTuple(
            loes_score_float,
            subject_session_uid,
            subject_str,
            session_str,
            session_date,
            is_validated
        )
        return candidate_info

    def test_get_mri_raw_candidate(self):
        candidate_info = self.get_test_candidate_info()
        mri_raw_candidate = get_mri_raw_candidate(candidate_info, True)
        self.assertIsNotNone(mri_raw_candidate)

    def test_get_mri_raw_candidate_training(self):
        candidate_info = self.get_test_candidate_info()
        mri_raw_candidate = get_mri_raw_candidate(candidate_info, False)
        self.assertIsNotNone(mri_raw_candidate)

    def test_loes_score_dataset_init(self):
        loes_score_dataset = LoesScoreDataset(scores_igor_updated_csv, val_stride=10, is_val_set_bool=False)
        self.assertIsNotNone(loes_score_dataset)
        length = loes_score_dataset.__len__()
        self.assertEqual(192, length)
        for i in range(3):
            item = loes_score_dataset.__getitem__(i)
            self.assertIsNotNone(item)
        item = loes_score_dataset.__getitem__(191)
        self.assertIsNotNone(item)


if __name__ == '__main__':
    unittest.main()
