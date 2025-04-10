import logging
import sys
import torchio as tio
import random


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)

class MRIAugmenter:
    """Class to handle MRI data augmentation"""
    
    def __init__(self, num_augmentations=3):
        self.num_augmentations = num_augmentations
    
    def augment(self, mri_tensor):
        """
        Generate augmented versions of an MRI tensor
        
        Args:
            mri_tensor: Original MRI tensor of shape [C, D, H, W]
            
        Returns:
            list: List of augmented tensors
        """
        augmented_tensors = []
        
        # Create a TorchIO Subject for augmentation
        subject = tio.Subject(image=tio.ScalarImage(tensor=mri_tensor))
        
        # Define augmentation transforms
        transforms = [
            # Spatial augmentations
            tio.RandomFlip(axes=('LR',)),
            tio.RandomAffine(scales=(0.9, 1.1), degrees=10),
            tio.RandomElasticDeformation(max_displacement=5),
            
            # Intensity augmentations  
            tio.RandomNoise(std=(0.01, 0.03)),
            tio.RandomGamma(log_gamma=(-0.3, 0.3))
        ]
        
        # Apply different combinations of transforms
        for _ in range(self.num_augmentations):
            # Randomly select 2-3 transforms
            num_transforms = random.randint(2, 3)
            selected_transforms = random.sample(transforms, num_transforms)
            
            # Create a composition of selected transforms
            transform = tio.Compose(selected_transforms)
            
            # Apply transform and add to list
            augmented_subject = transform(subject)
            augmented_tensors.append(augmented_subject.image.data)
        
        return augmented_tensors