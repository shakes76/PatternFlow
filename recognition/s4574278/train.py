import os

import torch
from torch import nn, optim
from torch.backends import cudnn
from torch.utils.data.dataset import random_split
from torchsummary import summary
from torch.cuda.amp import autocast, GradScaler
from dataset import IsicDataSet

from model import ModelSize, YOLOModel
from utils import yolox_loss


##########################################################
#   Constants
##########################################################

# where the cache saves
model_data_path = os.path.join("snapshot", "yolox.pth")

# for use we only have 1 class
classes = ["lesion"]
num_classes = len(classes)

# Must be multiple of 32
input_shape = (512, 512)

# Default Network Size
model_size: ModelSize = ModelSize.S

# Image Folder
image_folder = "dataset/input"
annotation_folder = "dataset/annotation"

# Suppress candidate boxes below this confidence
threshold = 0.5

# For NMS, the higher it goes, the less boxes it detect
iou = 0.8

# Turn on GPU or not
device = torch.device("cuda:0")

# How many CPU threads required
# tune up if CPU is the hurdle
# tune down if no enough ram
num_workers = 8

# Optimizer HP
lr = 1e-3
momentum = 0.937
weight_decay = 5e-4

# Batch size for training
batch_size = 64

# Max Epochs
max_epochs = 50


#####################################################
# Helpers
#####################################################


def _init_weight(module: nn.Module):
    classname = module.__class__.__name__
    if classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(module.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(module.bias.data, 0.0)
    elif (classname.find("Conv2d") != -1) and hasattr("weight"):
        torch.nn.init.xavier_normal_(module.weight.data, gain=0.3)


#####################################################
# Training
#####################################################


# Creates model and optimizer in default precision
model: nn.Module = YOLOModel(num_classes, model_size)
model.apply(_init_weight)
# read existing model data
if os.path.exists(model_data_path):
    model.load_state_dict(torch.load(model_data_path, map_location=device))

# Show model summary
summary(model, (3, 512, 512)).cuda()

# dataset
data = IsicDataSet(image_folder, annotation_folder, classes)
subset_train, subset_valid, subset_test = random_split(
    data, [2100, 234, 260], generator=torch.Generator().manual_seed(42)
)

train_loader = torch.utils.data.DataLoader(
    subset_train,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
    pin_memory=True,
)

valid_loader = torch.utils.data.DataLoader(
    subset_valid,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
    pin_memory=True,
)

# Optimizer
optimizer = optim.Adam(
    model.parameters(), lr=lr, betas=(momentum, 0.999), weight_decay=weight_decay
)
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
cudnn.benchmark = True

scaler = GradScaler()

for epoch in range(max_epochs):
    loss = 0
    steps = 0
    total_loss = 0
    for input, target in train_loader:
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward with autocast
        with autocast():
            output = model(input)
            loss = yolox_loss(output, target)
            total_loss = total_loss + loss
            steps = total_loss + 1

        # backward + optimize
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    val_loss = loss
    total_val_loss = 0
    val_steps = 0
    for input, target in valid_loader:
        with torch.no_grad():
            output = model(input)
            val_loss = yolox_loss(output, target)
            total_val_loss = total_val_loss + val_loss
            val_steps = val_steps + 1

    torch.save(
        model.state_dict(),
        f"logs/ep{epoch + 1:3d}-loss{total_loss/steps:3f}-val_loss{total_val_loss/val_steps:3f}.pth",
    )
