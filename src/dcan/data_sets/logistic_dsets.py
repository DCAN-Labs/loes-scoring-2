import copy
import functools
import logging
import os
import random

import torch
from dataclasses import dataclass, field
from torch.utils.data import Dataset
from typing import List

from util.disk import getCache

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

raw_cache = getCache('loes_score-4')

THRESHOLD = 0.1

@dataclass(order=True)
class CandidateInfoTuple:
    """Class for keeping track subject/session info."""
    loes_score_float: float
    file_path: str
    subject_str: str
    session_str: str
    cald_develops: int  
    augmentation_index: int = None
    sort_index: float = field(init=False, repr=False)

    def __hash__(self):
        return hash(self.file_path)

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


def get_candidate_info_list(folder, df, candidates: List[str]):
    candidate_info_list = []
    df = df.reset_index()  # make sure indexes pair with number of rows

    for _, row in df.iterrows():
        candidate = row['anonymized_subject_id']
        if candidate in candidates:
            append_candidate(folder, candidate_info_list, row)

    candidate_info_list.sort(reverse=True)

    return candidate_info_list


def append_candidate(folder, candidate_info_list, row):
    subject_str = row['anonymized_subject_id']
    session_str = row['anonymized_session_id']
    file_name = f"{subject_str}_{session_str}_space-MNI_brain_mprage_RAVEL.nii.gz"
    file_path = os.path.join(folder, file_name)
    loes_score_float = float(row['loes-score'])
    cald_develops = int(row['cald_develops'])
    candidate_info_list.append(CandidateInfoTuple(
        loes_score_float,
        file_path,
        subject_str,
        session_str,
        cald_develops
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
                 folder,
                 subjects: List[str], df, output_df,
                 is_val_set_bool=None,
                 subject=None,
                 sortby_str='random'
                 ):
        self.is_val_set_bool = is_val_set_bool
        self.candidateInfo_list = copy.copy(get_candidate_info_list(folder, df, subjects))

        if subject:
            self.candidateInfo_list = [
                x for x in self.candidateInfo_list if x.subject_str == subject
            ]

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
        if output_df is not None:
            for candidate_info in self.candidateInfo_list:
                row_location = (df["anonymized_subject_id"] == candidate_info.subject) & (df["anonymized_session_id"] == candidate_info.session_str)
                output_df.loc[row_location, 'training'] = 0 if is_val_set_bool else 1
                output_df.loc[row_location, 'validation'] = 1 if is_val_set_bool else 0

    def __len__(self):
        return len(self.candidateInfo_list)

    def __getitem__(self, ndx):
        candidate_info = self.candidateInfo_list[ndx]
        candidate_a = get_mri_raw_candidate(candidate_info, self.is_val_set_bool)
        candidate_t = candidate_a.to(torch.float32)

        loes_score = candidate_info.loes_score_float
        cald_develops = candidate_info.cald_develops
        cald_develops_t = torch.tensor(cald_develops, dtype=torch.float32)

        return candidate_t, cald_develops_t, candidate_info.subject_str, candidate_info.session_str
