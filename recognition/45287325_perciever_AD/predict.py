import torch
import torch.nn as nn
from torchsummary import summary

from modules import Perciever
from dataset import test_data_loader

classes = ('AD', 'NC')

MODEL_PATH = './perciever.pth'
DATA_PATH = "./Images/AD_NC"

# Model Parameters
NUM_LATENTS = 32
DIM_LATENTS = 128
NUM_CROSS_ATTENDS = 1
DEPTH_LATENT_TRANSFORMER = 4

net = Perciever(NUM_LATENTS, DIM_LATENTS, DEPTH_LATENT_TRANSFORMER, NUM_CROSS_ATTENDS)
net.load_state_dict(torch.load(MODEL_PATH))

correct = 0
total = 0

with torch.no_grad():
    for i, data in enumerate(test_data_loader(DATA_PATH, 4), 0):
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if i == 50:
            break

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

