import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader
from dataset import ADNI, dataset_dir
from train import print_accuracy

def predict(model, ds):
    """Classify a single image sequence and plot the images in scans_sequence.png."""
    loader = DataLoader(ds, shuffle=True, batch_size=1)
    seq, label = next(iter(loader))

    pred = model(seq)
    print(f"true label: {label.item()}")
    print(f"prediction: {pred.item()}")

    # Squeeze out batch and channel dim
    seq = seq.squeeze(2).squeeze(0)

    # Note: plotted images are cropped and standardized (differ from original)
    fig, axs = plt.subplots(4, 5, constrained_layout=True, figsize=(8, 6))
    fig.suptitle("Input sequence scans")
    for i, img in enumerate(seq):
        axs[i // 5][i % 5].imshow(img.cpu().numpy(), cmap="gray")
    plt.savefig("scans_sequence.png")

if __name__ == "__main__":
    batch_size = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load("model.pkl").to(device)
    ds = ADNI(device, dataset_dir)
    other_size  = int(np.round(ADNI.NUM_SEQUENCES * 0.8))
    ts_size  = int(np.round(ADNI.NUM_SEQUENCES * 0.2))
    ds, ts_ds = random_split(ds, (other_size, ts_size), torch.Generator().manual_seed(42))
    print_accuracy(model, ts_ds, batch_size)
    # predict(model, ts_ds)
