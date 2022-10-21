import os
import numpy as np
import nibabel as nib
import torch
from PIL import Image
from torch.utils.data import Dataset

def normalize(a):
    return 2.*(a - np.min(a))/np.ptp(a)-1

class OASISDataset(Dataset):
    """OASIS dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        # Find all files.
        self.files = [f for f in os.listdir(root_dir) if f.endswith(".nii.gz")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.files[idx])
        # Load NIFTI image using NiBabel
        nifti_image = nib.load(img_name)
        nii_data = nifti_image.get_fdata()
        # Normalize 0 mean unit variance.
        nii_data = normalize(nii_data)
        # Convert to PIL so can use torchvision transforms.
        image = Image.fromarray(nii_data)

        if self.transform:
            image = self.transform(image)

        return image