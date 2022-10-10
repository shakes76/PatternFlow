from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import os.path as osp
import glob
from PIL import Image
import random
import torch
import numpy as np
from torch.backends import cudnn


class Preprocessor(Dataset):
    def __init__(self, dataset, transformer, root):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.transformer = transformer
        self.root = root

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, label = self.dataset[index]
        img = Image.open(osp.join(self.root, fname)).convert('RGB')
        img = self.transformer(img)
        return img, label, index