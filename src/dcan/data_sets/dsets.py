import copy
import functools
import logging
import os
import random

import torch
import torchio as tio
from dataclasses import dataclass, field
from torch.utils.data import Dataset
from typing import List

from util.disk import getCache

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

raw_cache = getCache('loes_score-4')


@dataclass(order=True)
class CandidateInfoTuple:
    """Class for keeping track subject/session info."""
    loes_score_float: float
    file_path: str
    subject_str: str
    session_str: str
    has_ald: int
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
        session_str,
        row['has_ald']
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


# Add these imports to your dataset file
import torchio as tio
import torch

# Create augmentation transforms for training
def get_training_transforms():
    """Returns TorchIO transforms for training data augmentation"""
    return tio.Compose([
        # Spatial transforms
        tio.RandomAffine(
            scales=(0.9, 1.1),           # Scale by 90-110%
            degrees=(-10, 10),           # Rotate by ±10 degrees
            translation=(-5, 5),         # Translate by ±5mm
            p=0.5                        # Apply 50% of the time
        ),
        tio.RandomElasticDeformation(
            num_control_points=7,        # Control points for deformation
            max_displacement=7.5,        # Maximum displacement in mm
            p=0.3                        # Apply 30% of the time
        ),
        tio.RandomFlip(
            axes=('LR',),                # Only flip left-right (anatomically safe)
            p=0.5                        # Apply 50% of the time
        ),
        
        # Intensity transforms
        tio.RandomNoise(
            std=(0, 0.1),               # Add Gaussian noise
            p=0.3                       # Apply 30% of the time
        ),
        tio.RandomGamma(
            log_gamma=(-0.3, 0.3),      # Gamma correction
            p=0.3                       # Apply 30% of the time
        ),
        tio.RandomBiasField(
            coefficients=0.5,           # Simulate MRI bias field
            p=0.3                       # Apply 30% of the time
        ),
        tio.RandomBlur(
            std=(0, 1),                 # Gaussian blur
            p=0.2                       # Apply 20% of the time
        ),
        
        # Always apply normalization at the end
        tio.ZNormalization(masking_method=tio.ZNormalization.mean),
    ])

def get_validation_transforms():
    """Returns transforms for validation (no augmentation)"""
    return tio.Compose([
        tio.ZNormalization(masking_method=tio.ZNormalization.mean),
    ])

# Modify your LoesScoreDataset class to use augmentation
class LoesScoreDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None, augment=False):
        self.df = df
        self.transform = transform
        self.augment = augment
        
        # Set up transforms based on whether this is training or validation
        if augment:
            self.tio_transform = get_training_transforms()
        else:
            self.tio_transform = get_validation_transforms()
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load your image (adapt this to your current loading method)
        image_path = self.get_image_path(row)  # Your existing method
        
        # Load with TorchIO
        subject = tio.Subject(
            image=tio.ScalarImage(image_path),
            label=row['loes-score']  # Store label in subject
        )
        
        # Apply transforms
        transformed_subject = self.tio_transform(subject)
        
        # Extract tensor and label
        image_tensor = transformed_subject.image.data
        label = transformed_subject.label
        
        return image_tensor, torch.tensor(label, dtype=torch.float32), row['anonymized_subject_id'], row['anonymized_session_id']
    
    def __len__(self):
        return len(self.df)
    
    def get_image_path(self, row):
        subject = row['anonymized_subject_id']
        session = row['anonymized_session_id']
        return f'/path/to/images/{subject}_{session}_space-MNI_brain_mprage_RAVEL.nii.gz'
