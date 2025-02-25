import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, img_size=(512, 512)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.img_size = img_size
        self.filenames = self._get_valid_files()
        
    def _get_valid_files(self):
        valid_files = []
        for f in os.listdir(self.image_dir):
            if f.startswith('normalized_Image_'):
                mask_name = f"Mask_{f.replace('normalized_Image_', '')}"
                if os.path.exists(os.path.join(self.mask_dir, mask_name)):
                    valid_files.append(f)
                else:
                    print(f"Missing mask for image: {f}")
        return valid_files

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        base_name = img_name.replace('normalized_Image_', '')
        mask_name = f"Mask_{base_name}"
        
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # Load with checks
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found or corrupted: {img_path}")
            
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found or corrupted: {mask_path}")

        # Resize
        image = cv2.resize(image, self.img_size)
        mask = cv2.resize(mask, self.img_size)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']

        # Original normalization
        image = np.array(image, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
        mask = np.array(mask, np.float32)
        mask = np.expand_dims(mask, axis=0)  # Add channel dim
        mask = (mask > 0.5).astype(np.float32)  # Threshold

        return torch.tensor(image), torch.tensor(mask)