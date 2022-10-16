from dataset import *
from modules import model
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

MODE = "debug"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu")
model = model.to(device)
if MODE == "debug":
    print(device)

train_data = ISICDataset(
    image_folder_path="./data/ISIC-2017_Training_Data", 
    mask_folder_path="./data/ISIC-2017_Training_Part1_GroundTruth", 
    diagnoses_path="./data/ISIC-2017_Training_Part3_GroundTruth.csv",
    device=device,
    )
train_dataloader = torch.utils.data.DataLoader(
    train_data, 
    batch_size=2, 
    shuffle=True, 
    collate_fn=lambda x:list(zip(*x))
    )

def train_model(model, dataloader, n_epochs):
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    model.train()
    model.double()
    for epoch in tqdm(range(n_epochs)):
        for images, targets in tqdm(dataloader):
            optimizer.zero_grad()
            losses = model(images, targets)
            losses = sum(loss for loss in losses.values())
            losses.backward()
            optimizer.step()
        
            
train_model(model, train_dataloader, 10)