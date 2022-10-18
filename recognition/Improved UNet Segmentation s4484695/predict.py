import dataset
import modules
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

def test(dataLoaders, model, device):
    losses_validation = list()
    dice_similarities_validation = list()

    print("> Test")
    start = time.time()
    model.eval()
    with torch.no_grad():
        for images, labels in dataLoaders["valid"]:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            losses_validation.append(dice_loss(outputs, labels))
            dice_similarities_validation.append(dice_coefficient(outputs, labels))

        print('Validation Training Loss: {:.5f}, Validation Average Dice Similarity: {:.5f}'.format(get_average(losses_validation) ,get_average(dice_similarities_validation)))
    end = time.time()
    elapsed = end - start
    print("Validation took " + str(elapsed/60) + " mins in total")