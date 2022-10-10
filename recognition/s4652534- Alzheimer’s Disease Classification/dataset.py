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


class ADNI(Dataset):
    def __init__(self, dataset_dir):
        super(ADNI, self).__init__()
        self.dataset_dir = dataset_dir
        self.train_AD_dir = osp.join(osp.join(self.dataset_dir, 'train'), 'AD')
        self.train_NC_dir = osp.join(osp.join(self.dataset_dir, 'train'), 'NC')
        self.test_AD_dir = osp.join(osp.join(self.dataset_dir, 'test'), 'AD')
        self.test_NC_dir = osp.join(osp.join(self.dataset_dir, 'test'), 'NC')

        self.train_AD = self.process_dir(dir_path=self.train_AD_dir, label='AD')
        self.train_NC = self.process_dir(dir_path=self.train_NC_dir, label='NC')
        self.test_AD = self.process_dir(dir_path=self.test_AD_dir, label='AD')
        self.test_NC = self.process_dir(dir_path=self.test_NC_dir, label='NC')

    def process_dir(self, dir_path, label):
        img_paths = glob.glob(osp.join(dir_path, "*.jpeg"))
        data = []
        for img_path in img_paths:
            if label == "AD":
                data.append((img_path, 1))
            else:
                data.append((img_path, 0))
        return data