import torch
import torch.nn as nn
import torch.utils as utils
import torchvision

import modules
import dataset
device = "cuda" if torch.cuda.is_available() else "cpu"
data = dataset.DataLoader(
    r"C:\Users\blobf\COMP3710\PatternFlow\recognition\46425254-VQVAE\keras_png_slices_data\train")

data = utils.data.DataLoader(data, batch_size= 64, shuffle = True)


model = modules.VQVAE().to(device)
for i, _ in data:
    i = i.view(-1, 3, 256, 256).to(device)
    model(i)
    

