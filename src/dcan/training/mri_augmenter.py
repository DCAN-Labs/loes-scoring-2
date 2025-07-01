import logging
import sys
import random
import numpy as np
import torch
from scipy import ndimage

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
    """Class to handle MRI data augmentation without TorchIO"""
    
    def __init__(self, num_augmentations=3):
        self.num_augmentations = num_augmentations
    
    def random_flip_lr(self, image, prob=0.5):
        """Random left-right flip"""
        if np.random.random() < prob:
            flipped = np.flip(image, axis=-3)  # Assuming shape [C, D, H, W], flip D axis
            return flipped.copy()  # Make a copy to avoid negative strides
        return image
    
    def random_affine_transform(self, image, scale_range=(0.9, 1.1), rotation_degrees=10):
        """Random affine transformation using scipy"""
        # Remove channel dimension for processing
        if len(image.shape) == 4:  # [C, D, H, W]
            original_shape = image.shape
            image_3d = image[0]  # Take first channel
            process_3d = True
        else:
            image_3d = image
            process_3d = False
            
        # Random scaling
        scale = np.random.uniform(scale_range[0], scale_range[1])
        if scale != 1.0:
            zoom_factors = [scale, scale, scale]
            image_3d = ndimage.zoom(image_3d, zoom_factors, order=1)
            
            # Crop or pad to original size if needed
            if scale > 1.0:  # Image got bigger, crop center
                original_size = original_shape[-3:] if process_3d else image.shape
                start_indices = [(s - o) // 2 for s, o in zip(image_3d.shape, original_size)]
                end_indices = [start + o for start, o in zip(start_indices, original_size)]
                image_3d = image_3d[start_indices[0]:end_indices[0],
                                  start_indices[1]:end_indices[1], 
                                  start_indices[2]:end_indices[2]]
            elif scale < 1.0:  # Image got smaller, pad
                original_size = original_shape[-3:] if process_3d else image.shape
                pad_widths = [(max(0, (o - s) // 2), max(0, o - s - (o - s) // 2)) 
                             for s, o in zip(image_3d.shape, original_size)]
                image_3d = np.pad(image_3d, pad_widths, mode='constant', constant_values=0)
        
        # Random rotation
        angle = np.random.uniform(-rotation_degrees, rotation_degrees)
        if abs(angle) > 0.1:  # Only rotate if angle is significant
            # Randomly choose rotation axis
            axes_pairs = [(0, 1), (0, 2), (1, 2)]
            axes = random.choice(axes_pairs)
            image_3d = ndimage.rotate(image_3d, angle, axes=axes, reshape=False, order=1)
        
        # Restore channel dimension if needed
        if process_3d:
            return image_3d[np.newaxis, ...]  # Add channel dimension back
        return image_3d
    
    def random_elastic_deformation(self, image, max_displacement=5):
        """Simplified elastic deformation using random displacement fields"""
        if len(image.shape) == 4:  # [C, D, H, W]
            image_3d = image[0]
            process_3d = True
        else:
            image_3d = image
            process_3d = False
            
        # Create random displacement field
        shape = image_3d.shape
        # Create a coarse displacement field and then interpolate
        coarse_shape = [s // 4 for s in shape]  # 1/4 resolution
        
        # Random displacements for each dimension
        displacement_fields = []
        for dim in range(3):
            coarse_field = np.random.uniform(-max_displacement, max_displacement, coarse_shape)
            # Smooth the field
            coarse_field = ndimage.gaussian_filter(coarse_field, sigma=1.0)
            # Interpolate to full resolution
            full_field = ndimage.zoom(coarse_field, [s/c for s, c in zip(shape, coarse_shape)], order=1)
            displacement_fields.append(full_field)
        
        # Create coordinate grids
        coords = np.meshgrid(*[np.arange(s) for s in shape], indexing='ij')
        
        # Apply displacements
        new_coords = []
        for i, coord in enumerate(coords):
            new_coord = coord + displacement_fields[i]
            # Clamp to valid range
            new_coord = np.clip(new_coord, 0, shape[i] - 1)
            new_coords.append(new_coord)
        
        # Interpolate
        deformed = ndimage.map_coordinates(image_3d, new_coords, order=1, mode='nearest')
        
        if process_3d:
            return deformed[np.newaxis, ...]
        return deformed
    
    def add_noise(self, image, std_range=(0.01, 0.03)):
        """Add random Gaussian noise"""
        std = np.random.uniform(std_range[0], std_range[1])
        noise = np.random.normal(0, std, image.shape)
        return image + noise
    
    def random_gamma_correction(self, image, gamma_range=(-0.3, 0.3)):
        """Apply random gamma correction"""
        log_gamma = np.random.uniform(gamma_range[0], gamma_range[1])
        gamma = np.exp(log_gamma)
        
        # Normalize to [0, 1] for gamma correction
        img_min, img_max = image.min(), image.max()
        if img_max > img_min:
            normalized = (image - img_min) / (img_max - img_min)
            corrected = np.power(normalized, gamma)
            # Scale back to original range
            return corrected * (img_max - img_min) + img_min
        return image
    
    def augment(self, mri_tensor):
        """
        Generate augmented versions of an MRI tensor
        
        Args:
            mri_tensor: Original MRI tensor of shape [C, D, H, W] (torch.Tensor or numpy.ndarray)
            
        Returns:
            list: List of augmented tensors (same type as input)
        """
        # Convert to numpy if torch tensor
        is_torch = torch.is_tensor(mri_tensor)
        if is_torch:
            image_np = mri_tensor.cpu().numpy()
        else:
            image_np = mri_tensor.copy()
        
        # Ensure the input array is contiguous
        if not image_np.flags['C_CONTIGUOUS']:
            image_np = np.ascontiguousarray(image_np)
        
        augmented_tensors = []
        
        # Define available augmentations
        augmentation_functions = [
            lambda img: self.random_flip_lr(img, prob=0.5),
            lambda img: self.random_affine_transform(img),
            lambda img: self.random_elastic_deformation(img),
            lambda img: self.add_noise(img),
            lambda img: self.random_gamma_correction(img)
        ]
        
        # Apply different combinations of transforms
        for _ in range(self.num_augmentations):
            # Start with original image
            augmented_img = image_np.copy()
            
            # Randomly select 2-3 transforms
            num_transforms = random.randint(2, 3)
            selected_functions = random.sample(augmentation_functions, num_transforms)
            
            # Apply selected transforms sequentially
            for transform_func in selected_functions:
                try:
                    augmented_img = transform_func(augmented_img)
                    # Ensure the result is contiguous after each transform
                    if isinstance(augmented_img, np.ndarray) and not augmented_img.flags['C_CONTIGUOUS']:
                        augmented_img = np.ascontiguousarray(augmented_img)
                except Exception as e:
                    log.warning(f"Augmentation failed: {e}, skipping this transform")
                    continue
            
            # Final check: ensure the array is contiguous and doesn't have negative strides
            if isinstance(augmented_img, np.ndarray):
                if not augmented_img.flags['C_CONTIGUOUS'] or any(s < 0 for s in augmented_img.strides):
                    augmented_img = np.ascontiguousarray(augmented_img)
            
            # Convert back to torch tensor if input was torch tensor
            if is_torch:
                augmented_img = torch.from_numpy(augmented_img)
            
            augmented_tensors.append(augmented_img)
        
        return augmented_tensors