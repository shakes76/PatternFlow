import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

def img_filename_as_numerical(s: str):
    seq_id, img_idx = s[:-5].split("/")[-1].split("_")
    return (int(seq_id), int(img_idx))

class ADNI(Dataset):
    """
    This dataset treats the ADNI data as fixed-length sequences of brain scans. These brain scans
    are images with a single color channel and a dimension of 240 x 256. The shape of a resulting
    batch tensor is (B, S, C, H, W) where B = batch size, S = sequence length (20),
    C = color channels (1), H = height (240) and W = width (256).
    """
    NUM_SEQUENCES = 1526	# The number of image sequences in the dataset
    SEQUENCE_LENGTH = 20	# The length of each image sequence
    C = 1					# Image channels
    H = 240					# Image height
    W = 256					# Image width

    def __init__(self, device, dataset_dir):
        self._device = device
        self._dataset_dir = dataset_dir

        # List of image sequences and labels
        self._sequences: list[tuple[list[str], int]] = []

        # Collect image sequences into self._sequences
        for subset in ("train", "test"):
            for label, class_dir in enumerate(("NC", "AD")):
                seq_prev = None

                img_paths = os.listdir(os.path.join(dataset_dir, subset, class_dir))
                img_paths.sort(key=img_filename_as_numerical)
                for img_path in img_paths:
                    seq_curr, _ = img_path.split("_")

                    if seq_prev != seq_curr:
                        self._sequences.append(([], label))

                    self._sequences[-1][0].append(os.path.join(subset, class_dir, img_path))
                    seq_prev = seq_curr

        assert len(self._sequences) == ADNI.NUM_SEQUENCES
        for s, _ in self._sequences:
            assert len(s) == ADNI.SEQUENCE_LENGTH

    def __len__(self):
        return len(self._sequences)

    def __getitem__(self, idx):
        seq, label = self._sequences[idx]
        t = torch.zeros((ADNI.SEQUENCE_LENGTH, ADNI.C, ADNI.H, ADNI.W), device=self._device)
        for i, img_path in enumerate(seq):
            t[i] = read_image(os.path.join(self._dataset_dir, img_path))

        return t, torch.tensor(label, dtype=torch.float32, device=self._device).unsqueeze(0)

if __name__ == "__main__":
    device = torch.device("cpu")
    ds = ADNI(device, "/home/jingweini/Documents/uni/COMP3710/report/data/AD_NC")
    seq, label = ds[0]
    print(f"seq.shape  : {seq.shape}")
    print(f"label.shape: {label.shape}")
