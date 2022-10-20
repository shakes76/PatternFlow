import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
from dataset import ADNI
from modules import AdniClassifier

def print_accuracy(model, ds, batch_size):
    loader = DataLoader(ds, batch_size=batch_size)
    loop = tqdm(loader)
    correct = 0
    for batch, label in loop:
        pred = model(batch)
        pred = torch.where(pred > 0.5, 1, 0).to(torch.int8)
        correct += torch.eq(pred, label.to(torch.int8)).sum()
    print(f"accuracy: {(correct / len(ds)).item():.5f}")

def train(device, tr_ds, batch_size, epochs=24):
    loader = DataLoader(tr_ds, shuffle=True, batch_size=batch_size)
    model = AdniClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), 1e-5)
    criterion = nn.BCELoss()

    batch_losses = []
    loop = tqdm(range(epochs))
    for epoch in loop:
        batch_loss = 0
        for batch, label in loader:
            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            batch_loss += loss.item()
        batch_losses.append(batch_loss)
        loop.set_postfix(batch_loss=f"{batch_loss:.5f}")

    return model, batch_losses

if __name__ == "__main__":
    batch_size = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = ADNI(device, "/home/jingweini/Documents/uni/COMP3710/report/data/AD_NC")

    tr_size  = int(np.round(ADNI.NUM_SEQUENCES * 0.65))
    val_size = int(np.round(ADNI.NUM_SEQUENCES * 0.15))
    ts_size  = int(np.round(ADNI.NUM_SEQUENCES * 0.2))
    print(f"train set size:      {tr_size}")
    print(f"validation set size: {val_size}")
    print(f"test set size:       {ts_size}")
    print(f"total set size:      {tr_size + val_size + ts_size}")
    ds, ts_ds = random_split(ds, (tr_size + val_size, ts_size), torch.Generator().manual_seed(42))
    tr_ds, val_ds = random_split(ds, (tr_size, val_size))

    model, batch_losses = train(device, tr_ds, batch_size)

    torch.save(model, "model.pkl")
    plt.plot(batch_losses)
    plt.title("Batch Loss")
    plt.xlabel("Epoch")
    plt.savefig("batch_losses.png")

    print_accuracy(model, val_ds, batch_size)
