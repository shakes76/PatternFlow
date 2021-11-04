from pathlib import Path

import torch
from torch import optim
from torch.backends import cudnn
from torch.utils.data.dataset import random_split
from torch.cuda.amp import autocast, GradScaler
import wandb
from data import IsicDataSet
from model import ModelSize, YoloxModel
from loss import YoloxLoss

##########################################################
#   Constants
##########################################################

# where the cache saves
model_data_folder = Path("snapshot")
model_data_path = model_data_folder / "yolox.pth"

# for use we only have 1 class
classes = ["lesion"]

# Must be multiple of 32
input_shape = (512, 512)

# Default Network Size
model_size: ModelSize = ModelSize.S

# Image Folder
image_folder = Path("dataset") / "input"
annotation_folder = Path("dataset") / "annotation"

# Suppress candidate boxes below this confidence
threshold = 0.5

# For NMS, the higher it goes, the less boxes it detect
iou = 0.8

# Turn on GPU or not
device = torch.device('cuda')

# How many CPU threads required
# default 4 times of # of GPU
num_workers = 4

# Optimizer HP
lr = 1e-3

# Batch size for training
batch_size = 64

# Max Epochs
max_epochs = 50

#####################################################
# Training
#####################################################
# Creates model and optimizer in default precision
model = YoloxModel(classes, model_size).cuda()
# If model_data_path is not supplied, or prefer train from scratch
# leave it as None, not ""
model_data_path = None
model.init_state(model_data_path, map_location=device)

# Show model summary
wandb.init(project="pattern", entity="parisq")
wandb.watch(model)
wandb.config = {
  "learning_rate": lr,
  "epochs": max_epochs,
  "batch_size": batch_size
}
# summary(model, (3, 512, 512)).cuda()

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
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
# lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=max_epochs, steps_per_epoch=len(train_loader))

# Higher performance
cudnn.benchmark = True

# Mixed precision
scaler = GradScaler()

for epoch in range(max_epochs):
    loss = 0
    steps = 0
    total_loss = 0
    for input, target in train_loader:
        # clear the parameter gradients
        optimizer.zero_grad()

        # load data
        with torch.no_grad():
            input = input.type(torch.half).cuda()
            target = [bboxes.type(torch.half).cuda() for bboxes in target]

        # forward with autocast for Mixed precision
        with autocast():
            output = model(input)
            loss = YoloxLoss(output, target)
            total_loss = total_loss + loss
            steps = total_loss + 1

        # backward + optimize
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    val_loss = 0
    total_val_loss = 0
    val_steps = 0
    for input, target in valid_loader:
        with torch.no_grad():
            output = model(input)
            val_loss = YoloxLoss(output, target)
            total_val_loss = total_val_loss + val_loss
            val_steps = val_steps + 1

    model.save_state(
        model_data_folder / f"ep{epoch + 1:3d}-loss{total_loss / steps:3f}-val_loss{total_val_loss / val_steps:3f}.pth")
