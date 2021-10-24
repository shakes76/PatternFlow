import os
import numpy as np
import nibabel as nib
import torch
from torch import Tensor
from PIL import Image
import torch.nn.functional as F
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
        self.files = [f for f in os.listdir(root_dir) if f.endswith(".nii.gz")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.files[idx])
        nifti_image = nib.load(img_name)
        nii_data = nifti_image.get_fdata()
        nii_data = normalize(nii_data)
        image = Image.fromarray(nii_data)

        if self.transform:
            image = self.transform(image)

        return image

    # class Resize(object):
    #     """ Resize the volume to a given size."""

    #     def __init__(self, output_size):
    #         assert isinstance(output_size, (int))
    #         self.output_size = output_size

    #     def __call__(self, x: Tensor):
    #         d = torch.linspace(-1, 1, self.output_size)
    #         meshx, meshy, meshz = torch.meshgrid((d, d, d), indexing='ij')
    #         grid = torch.stack((meshx, meshy, meshz), 3)
    #         grid = grid.unsqueeze(0) # add batch dim
    #         x = x.unsqueeze(0) # add batch dim
    #         x = F.grid_sample(x, grid, align_corners=True)
    #         x = x.squeeze(0)
    #         return x

    # class ToTensor(object):
    #     def __call__(self, x):
    #         return torch.Tensor(x)