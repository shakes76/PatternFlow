import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm

def predict(model, test_loader, device):
    """Predict the labels for the test set.

    Args:
        model (nn.Module): The model to be evaluated.
        test_loader (DataLoader): The test set dataloader.
        device (str): The device to be used for training.

    Returns:
        list: The predicted labels for the test set.
    """
    model.eval()
    predictions = []
    for batch in tqdm(test_loader):
        images, labels = batch
        with torch.no_grad():
            logits = model(images.to(device))
        predictions.append(logits.argmax(dim=-1).cpu().numpy())
    print("pred:",predictions,"\nlabels:",labels)
