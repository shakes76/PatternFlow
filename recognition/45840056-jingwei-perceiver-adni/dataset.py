import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

class ADNIDataset(Dataset):
    NUM_SCAN_GROUPS = 1526
    SCAN_GROUP_SIZE = 20
    H = 240
    W = 256

    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self._scan_groups = []

        for subset in ("train", "test"):
            for label, class_dir in enumerate(("NC", "AD")):
                scan_group_prev = None
                for img_path in os.listdir(os.path.join(dataset_dir, subset, class_dir)):
                    scan_group_curr, _ = img_path.split("_")

                    if scan_group_prev != scan_group_curr:
                        self._scan_groups.append(([], label))

                    self._scan_groups[-1][0].append(os.path.join(subset, class_dir, img_path))
                    scan_group_prev = scan_group_curr

        assert len(self._scan_groups) == ADNIDataset.NUM_SCAN_GROUPS
        for g, label in self._scan_groups:
            assert len(g) == ADNIDataset.SCAN_GROUP_SIZE

    def __len__(self):
        return len(self._scan_groups)

    def __getitem__(self, idx):
        scans, label = self._scan_groups[idx]
        img_seq = torch.zeros(ADNIDataset.SCAN_GROUP_SIZE, ADNIDataset.H, ADNIDataset.W)
        for i, img_path in enumerate(scans):
            img_seq[i] = read_image(os.path.join(self.dataset_dir, img_path))

        return img_seq, label

if __name__ == "__main__":
    ds = ADNIDataset("/home/jingweini/Documents/uni/COMP3710/report/data/AD_NC")
