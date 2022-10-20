import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.io import read_image

dataset_dir = "/home/jingweini/Documents/uni/COMP3710/report/data/AD_NC"

def img_filename_as_numerical(s: str):
    """Create key from image file name for sorting"""
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
    H = 224					# Image height
    W = 224					# Image width

    def __init__(self, device, dataset_dir):
        self._device = device
        self._dataset_dir = dataset_dir
        self.transform = transforms.Compose((
            transforms.CenterCrop(224),
            # Calculated using the calculate_dataset_mean_stdev function
            transforms.Normalize((0.1419018656015396,), (0.24209415912628174,))
        ))

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
        # Create sequence tensor
        t = torch.zeros((ADNI.SEQUENCE_LENGTH, ADNI.C, ADNI.H, ADNI.W), device=self._device)
        for i, img_path in enumerate(seq):
            t[i] = self.transform(read_image(os.path.join(self._dataset_dir, img_path)) / 255.0)

        return t, torch.tensor(label, dtype=torch.float32, device=self._device).unsqueeze(0)

def calculate_dataset_mean_stdev(loader, batch_size):
    """Calculate the mean and stdev for each channel over the entire dataset."""
    sum = sum_squared = num_batches = 0
    for batch, _ in loader:
        batch = batch.reshape(batch_size * ADNI.SEQUENCE_LENGTH, 1, ADNI.H, ADNI.W)
        sum += torch.mean(batch, dim=(0, 2, 3))
        sum_squared += torch.mean(batch ** 2, dim=(0, 2, 3))
        num_batches += 1
    mean = sum / num_batches
    # stdev = sqrt(E[X^2] - E[X]^2)
    std = torch.sqrt(sum_squared / num_batches - mean ** 2)
    return mean, std

if __name__ == "__main__":
    device = torch.device("cpu")
    ds = ADNI(device, dataset_dir)
    batch_size = 14

    # loader = DataLoader(ds, batch_size=batch_size)
    # mean, stdev = calculate_dataset_mean_stdev(loader, batch_size)
    # print(f"mean: {mean.item()}")
    # print(f"stdev: {stdev.item()}")

    # loader = DataLoader(ds, shuffle=True, batch_size=batch_size)
    # seq, label = next(iter(loader))
    # print(f"seq.shape  : {seq.shape}")
    # print(f"label.shape: {label.shape}")
    # seq = seq[0]
    # label = label[0]

    # for i, img in enumerate(seq):
    #     plt.imshow(img.squeeze(0), cmap="gray")
    #     plt.savefig(f"{i}.png")
