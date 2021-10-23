import argparse
import os
import nibabel as nib
import torch
from torch.utils.data import Dataset
from dotenv import load_dotenv

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=int, default=20, help="The percentage of data to hold aside for the test set. Default=20.")
args = parser.parse_args()
load_dotenv()

data_path = os.getenv("OASIS_PATH")

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
        self.files = [f for f in os.listdir(data_path) if f.endswith(".nii.gz")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.files[idx])
        nifti_image = nib.load(img_name)
        nii_data = nifti_image.get_fdata()

        if self.transform:
            nii_data = self.transform(nii_data)

        return nii_data