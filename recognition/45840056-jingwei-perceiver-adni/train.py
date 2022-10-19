import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import ADNIDataset
from modules import ADNIClassifier

def train(device, epochs=20):
    batch_losses = []
    loader = DataLoader(ADNIDataset("/home/jingweini/Documents/uni/COMP3710/report/data/AD_NC", device),\
        shuffle=True, batch_size=4)
    model = ADNIClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), 1e-5)
    criterion = nn.BCELoss()

    loop = tqdm(range(epochs))
    for epoch in loop:
        batch_loss = 0
        for batch, label in loader:
            optimizer.zero_grad()
            pred = model(batch).squeeze(-1)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            batch_loss += loss.item()
        batch_losses.append(batch_loss)
        loop.set_postfix(batch_loss=f"{batch_loss:.5f}")

    return model, batch_losses

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, batch_losses = train(device)
    torch.save(model, "modelv0.1.pkl")
    plt.plot(batch_losses)
    plt.savefig("batch_losses.png")
