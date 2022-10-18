
from vit_pytorch import ViT
import argparse
import torchvision.transforms as transforms
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import os
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import BrainDataset
from modules import ViT
from train import train, test
from predict import predict


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

torch.manual_seed(42)


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--dim', type=int, default=512,
                    help='neural network dimension')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate of the optimizer')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--depth', type=int, default=8,
                    help='depth of the network')
parser.add_argument('--heads', type=int, default=16,
                    help='number of heads in the multihead attention')
parser.add_argument('--epochs', type=int, default=500,
                    help='number of epochs to train')
parser.add_argument('--mlp_dim', type=int, default=1024,
                    help='dimension of the mlp on top of the attention block')
parser.add_argument('--model', type=str, default='ViT',
                    help='ViT or SepViT')
parser.add_argument('--mode', type=str, default='train', help='dropout rate')
opt = parser.parse_args()


image_size = 224

train_tfm = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(0),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.2641372), std=(0.5060895))
])
test_tfm = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.2641372), std=(0.5060895))
])
_dataset_dir = "./datasets/AD_NC/"


device = "cuda" if torch.cuda.is_available() else "cpu"

model = ViT(
    image_size=image_size,
    patch_size=16,
    num_classes=2,
    dim=opt.dim,
    depth=opt.depth,
    heads=opt.heads,
    mlp_dim=opt.mlp_dim,
    dropout=0.2,
    emb_dropout=0.2,
    channels=1
).to(device)

from vit_pytorch.sep_vit import SepViT

# model = SepViT(
#     num_classes =2,
#     dim = 64,               # dimensions of first stage, which doubles every stage (32, 64, 128, 256) for SepViT-Lite
#     dim_head = opt.heads,          # attention head dimension
#     heads = (1, 2, 4, 8),   # number of heads per stage
#     depth = (1, 2, 6, 2),   # number of transformer blocks per stage
#     window_size = 7,        # window size of DSS Attention block
#     dropout = 0.2    ,       # dropout,
#     channels=1
# ).to(device)
# model.load_state_dict(torch.load('./SepVit_64_dp02_da_new.ckpt'))

if opt.mode == 'train':

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=opt.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=16, T_mult=1)

    train_set = BrainDataset(os.path.join(
        _dataset_dir, "train"), tfm=train_tfm, split='train')
    train_loader = DataLoader(
        train_set, batch_size=opt.batch_size, shuffle=True, pin_memory=True)

    val_set = BrainDataset(os.path.join(
        _dataset_dir, "train"), tfm=test_tfm, split='val')
    val_loader = DataLoader(
        val_set, batch_size=opt.batch_size, shuffle=True, pin_memory=True)

    test_set = BrainDataset(os.path.join(_dataset_dir, "test"), tfm=test_tfm)
    test_loader = DataLoader(test_set, batch_size=opt.batch_size,
                             shuffle=True, pin_memory=True)
    train(model, train_loader, val_loader, optimizer, scheduler,
          criterion, epochs=opt.epochs, writer=writer, device=device,test_loader=test_loader)
    test(model, test_loader, device=device)
if opt.mode == 'test':
    model.load_state_dict(torch.load('./pretrained_model.ckpt'))


    test_set = BrainDataset(os.path.join(_dataset_dir, "test"), tfm=test_tfm)
    test_loader = DataLoader(test_set, batch_size=opt.batch_size,
                             shuffle=True, pin_memory=True)

    test(model, test_loader, device=device)

elif opt.mode == 'pred':
    model.load_state_dict(torch.load('./pretrained_model.ckpt'))
    pred_set = BrainDataset(os.path.join(
        _dataset_dir, "test"), tfm=test_tfm, split='pred')
    pred_loader = DataLoader(pred_set, batch_size=opt.batch_size,
                             shuffle=True, pin_memory=True)
    predict(model, pred_loader, device=device)
  

