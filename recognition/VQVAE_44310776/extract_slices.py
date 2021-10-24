import os
import numpy as np
import nibabel as nib
import torch
from torch import Tensor
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

root_dir = os.getenv("OASIS_PATH")
files = [f for f in os.listdir(root_dir) if f.endswith(".nii.gz")]

for f in tqdm(os.listdir(root_dir)):
    if not f.endswith(".nii.gz"):
        continue

    img_name = os.path.join(root_dir, f)
    nifti_image = nib.load(img_name)
    nii_data = nifti_image.get_fdata()
    midpoint = nii_data.shape[0] // 2
    slice = nii_data[midpoint]
    slice_image = nib.Nifti1Image(slice, np.eye(4))
    filename = os.path.join("./data/oasis_slices", f)
    slice_image.to_filename(filename)