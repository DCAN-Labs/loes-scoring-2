import copy
import functools
import logging
import os
import random

import torch
from dataclasses import dataclass, field
from torch.utils.data import Dataset
from typing import List
from torch.utils.data import DataLoader
import nibabel as nib
import numpy as np
from scipy import ndimage


from util.disk import getCache

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

raw_cache = getCache('loes_score-3')


@dataclass(order=True)
class CandidateInfoTuple:
    """Class for keeping track subject/session info."""
    loes_score_float: float
    file_path: str
    subject_str: str
    session_str: str
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
    candidate_info_list.append(CandidateInfoTuple(
        loes_score_float,
        file_path,
        subject_str,
        session_str
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


def z_normalize(image, mask=None):
    """Z-normalization (standardization) of image data"""
    if mask is not None:
        masked_data = image[mask > 0]
        mean = np.mean(masked_data)
        std = np.std(masked_data)
    else:
        mean = np.mean(image)
        std = np.std(image)
    
    if std == 0:
        return image - mean
    return (image - mean) / std


def random_flip_lr(image, prob=0.5):
    """Random left-right flip"""
    if np.random.random() < prob:
        return np.flip(image, axis=0)  # Assuming first axis is left-right
    return image


def random_affine_transform(image, prob=0.8):
    """Simple random affine transformation using scipy"""
    if np.random.random() > prob:
        return image
    
    # Small random rotation (in degrees)
    angle = np.random.uniform(-5, 5)
    
    # Small random translation
    translation = [np.random.uniform(-2, 2) for _ in range(3)]
    
    # Apply rotation around center
    for axis in [(0, 1), (0, 2), (1, 2)]:
        if np.random.random() < 0.3:  # 30% chance for each axis pair
            image = ndimage.rotate(image, angle, axes=axis, reshape=False, order=1)
    
    # Apply translation
    image = ndimage.shift(image, translation, order=1)
    
    return image


class LoesScoreMRIs:
    def __init__(self, candidate_info, is_val_set_bool):
        mprage_path = candidate_info.path_to_file
        
        # Load NIfTI file using nibabel instead of TorchIO
        nii_img = nib.load(mprage_path)
        image_data = nii_img.get_fdata()
        
        # Convert to float32 and ensure it's a numpy array
        image_data = np.array(image_data, dtype=np.float32)
        
        # Z-normalization (equivalent to tio.ZNormalization)
        image_data = z_normalize(image_data)
        
        # Apply augmentations only for training
        if not is_val_set_bool:
            # Random left-right flip (equivalent to tio.RandomFlip(axes='LR'))
            image_data = random_flip_lr(image_data)
            
            # Random affine or elastic deformation (simplified version)
            if np.random.random() < 0.8:
                image_data = random_affine_transform(image_data)
            # Note: Elastic deformation is more complex without TorchIO
        
        # Convert to torch tensor and add channel dimension
        self.mprage_image_tensor = torch.from_numpy(image_data.copy()).unsqueeze(0)  # Add channel dim
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
        loes_score_t = torch.tensor(loes_score, dtype=torch.float32)

        return candidate_t, loes_score_t, candidate_info.subject_str, candidate_info.session_str