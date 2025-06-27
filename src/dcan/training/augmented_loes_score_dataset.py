import copy
import logging
import sys

import torch
from dcan.data_sets.logistic_dsets import LoesScoreDataset, get_mri_raw_candidate
from dcan.training.mri_augmenter import MRIAugmenter



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)


class AugmentedLoesScoreDataset(LoesScoreDataset):
    """Dataset with augmentation for minority class samples"""
    
    def __init__(self, folder, subjects, df, output_df, is_val_set_bool=None, 
                 augment_minority=False, num_augmentations=3, **kwargs):
        super().__init__(folder, subjects, df, output_df, is_val_set_bool, **kwargs)
        
        self.augment_minority = augment_minority and not is_val_set_bool  # Only augment training set
        
        if self.augment_minority:
            # Find minority class samples
            minority_candidates = []
            
            for candidate in self.candidateInfo_list:
                if not candidate.cald_develops:  
                    minority_candidates.append(candidate)
            
            log.info(f"Found {len(minority_candidates)} minority class samples in dataset")
            
            # Create augmenter
            self.augmenter = MRIAugmenter(num_augmentations)
            
            # Create augmented versions of minority samples
            self.augmented_candidates = []
            
            for candidate in minority_candidates:
                # Load original MRI
                mri_tensor = get_mri_raw_candidate(candidate, is_val_set_bool)
                
                # Create augmented versions
                augmented_tensors = self.augmenter.augment(mri_tensor)
                
                # Create new candidate info tuples for augmented samples
                for i, aug_tensor in enumerate(augmented_tensors):
                    # Create a copy of the original candidate info
                    aug_candidate = copy.deepcopy(candidate)
                    aug_candidate.augmentation_index = i
                    self.augmented_candidates.append((aug_candidate, aug_tensor))
            
            log.info(f"Created {len(self.augmented_candidates)} augmented minority samples")
    
    def __len__(self):
        if self.augment_minority:
            return len(self.candidateInfo_list) + len(self.augmented_candidates)
        else:
            return len(self.candidateInfo_list)
    
    def __getitem__(self, idx):
        # Return original sample
        if idx < len(self.candidateInfo_list):
            return super().__getitem__(idx)
        
        # Return augmented sample
        else:
            aug_idx = idx - len(self.candidateInfo_list)
            aug_candidate, aug_tensor = self.augmented_candidates[aug_idx]
            
            # Convert tensor to correct type
            aug_tensor = aug_tensor.to(torch.float32)
            
            # Create label
            cald_develops = torch.tensor(0.0, dtype=torch.float32)  # Always 0 for minority class
            
            return aug_tensor, cald_develops, aug_candidate.subject_str, aug_candidate.session_str