import dataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import time

# Hyper-parameters
num_epochs = 35
learning_rate = 0.1
batchSize = 4

validDataSet = dataset.ISIC2017DataSet(r"C:\Users\kamra\OneDrive\Desktop\Uni Stuff\2022\COMP3710\Report\Small Data\Validation\Images", r"C:\Users\kamra\OneDrive\Desktop\Uni Stuff\2022\COMP3710\Report\Small Data\Validation\Labels", dataset.ISIC_Transform_Valid())

train_dataloader = DataLoader(validDataSet, batch_size=batchSize, shuffle=False)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

_, axs = plt.subplots(batchSize, 2)
for row in range(batchSize):
    img = train_features[row]
    img = img.permute(1,2,0).numpy()
    label = train_labels[row]
    label = label.permute(1,2,0).numpy()
    axs[row][0].imshow(img)
    axs[row][1].imshow(label, cmap="gray")

plt.show()