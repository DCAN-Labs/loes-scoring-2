import copy
import csv
import functools
import logging
import os
import random
from dataclasses import dataclass, field
from datetime import datetime

import torch
import torchio as tio
from torch.utils.data import Dataset

from util.disk import getCache

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

raw_cache = getCache('dcan_loes_score')


@dataclass(order=True)
class CandidateInfoTuple:
    """Class for keeping track subject/session info."""
    loes_score_float: float
    subject_session_uid: str
    subject_str: str
    session_str: str
    file_path: str
    session_date: datetime
    augmentation_index: int = None
    sort_index: float = field(init=False, repr=False)

    def __hash__(self):
        return hash(self.subject_session_uid)

    @property
    def subject(self) -> str:
        return self.subject_str

    def __post_init__(self):
        # sort by Loes score
        self.sort_index = self.loes_score_float

    @property
    def path_to_file(self) -> str:
        return self.file_path


def get_subject(p):
    return os.path.split(os.path.split(os.path.split(p)[0])[0])[1][4:]


def get_session(p):
    return os.path.split(os.path.split(p)[0])[1][4:]


def get_uid(p):
    return f'{get_subject(p)}_{get_session(p)}'


@functools.lru_cache(1)
def get_candidate_info_list(scores_csv, include_gd_only=False):
    candidate_info_list = []
    with open(scores_csv, "r") as f:
        for row in list(csv.reader(f))[1:]:
            is_gd = int(row[1])
            if not include_gd_only and not is_gd:
                append_candidate(candidate_info_list, row)
            elif include_gd_only and is_gd:
                append_candidate(candidate_info_list, row)

    candidate_info_list.sort(reverse=True)

    return candidate_info_list


def append_candidate(candidate_info_list, row):
    file_path = row[0]
    ses_pos = file_path.find('ses-')
    date_str = file_path[ses_pos + 4:ses_pos + 12]
    session_date = datetime.strptime(date_str, '%Y%m%d')
    loes_score_float = float(row[2])
    sub_start_pos = file_path.find('sub-')
    subject_session_uid = file_path[sub_start_pos:sub_start_pos + 25]
    subject_str, session_str = subject_session_uid.split('_')
    candidate_info_list.append(CandidateInfoTuple(
        loes_score_float,
        subject_session_uid,
        subject_str,
        session_str,
        file_path,
        session_date
    ))


def get_subject_session_info(row, partial_loes_scores, anatomical_region):
    subject_session_uid = row[1].strip()
    pos = subject_session_uid.index('_')
    session_str = subject_session_uid[pos + 1:]
    subject_str = row[0]
    session = partial_loes_scores[subject_str][subject_session_uid]
    if anatomical_region == 'ParietoOccipitalWhiteMatter':
        loes_score = session.parieto_occipital_white_matter.get_score()
    elif anatomical_region == 'all':
        loes_score = session.loes_score
    else:
        assert False

    return session_str, subject_session_uid, subject_str, loes_score


class LoesScoreMRIs:
    def __init__(self, candidate_info, is_val_set_bool):
        mprage_path = candidate_info.path_to_file
        mprage_image = tio.ScalarImage(mprage_path)
        if is_val_set_bool:
            transform = tio.Compose([
                tio.ToCanonical(),
                tio.ZNormalization(masking_method=tio.ZNormalization.mean),
            ])
        else:
            transform = tio.Compose([
                tio.ToCanonical(),
                tio.ZNormalization(masking_method=tio.ZNormalization.mean),
                tio.RandomFlip(axes='LR'),
                tio.OneOf({
                    tio.RandomAffine(): 0.8,
                    tio.RandomElasticDeformation(): 0.2,
                })
            ])
        transformed_mprage_image = transform(mprage_image)
        self.mprage_image_tensor = transformed_mprage_image.data

        self.subject_session_uid = candidate_info

    def get_raw_candidate(self):
        return self.mprage_image_tensor


@functools.lru_cache(1, typed=True)
def get_loes_score_mris(candidate_info, is_val_set_bool):
    return LoesScoreMRIs(candidate_info, is_val_set_bool)


@raw_cache.memoize(typed=True)
def get_mri_raw_candidate(subject_session_uid, is_val_set_bool):
    loes_score_mris = get_loes_score_mris(subject_session_uid, is_val_set_bool)
    mprage_image_tensor = loes_score_mris.get_raw_candidate()

    return mprage_image_tensor


class LoesScoreDataset(Dataset):
    def __init__(self,
                 cvs_data_file,
                 val_stride=0,
                 is_val_set_bool=None,
                 subject=None,
                 sortby_str='random',
                 use_gd_only=False
                 ):
        self.is_val_set_bool = is_val_set_bool
        self.candidateInfo_list = copy.copy(get_candidate_info_list(cvs_data_file, use_gd_only))

        if subject:
            self.candidateInfo_list = [
                x for x in self.candidateInfo_list if x.subject_str == subject
            ]

        if is_val_set_bool:
            assert val_stride > 0, val_stride
            self.candidateInfo_list = self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list
        elif val_stride > 0:
            del self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list

        if sortby_str == 'random':
            random.shuffle(self.candidateInfo_list)
        elif sortby_str == 'loes_score':
            pass
        else:
            raise Exception("Unknown sort: " + repr(sortby_str))

        log.info("{!r}: {} {} samples".format(
            self,
            len(self.candidateInfo_list),
            "validation" if is_val_set_bool else "training",
        ))

    def __len__(self):
        return len(self.candidateInfo_list)

    def __getitem__(self, ndx):
        candidate_info = self.candidateInfo_list[ndx]
        # TODO Possibly handle other file types such as diffusion-weighted sequences
        candidate_a = get_mri_raw_candidate(candidate_info, self.is_val_set_bool)
        candidate_t = candidate_a.to(torch.float32)

        loes_score = candidate_info.loes_score_float
        loes_score_t = torch.tensor(loes_score)

        return candidate_t, loes_score_t
