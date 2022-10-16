from dataset import ISICDataset
from modules import get_model
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

MODE = "debug"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = get_model()
model = model.to(device)
if MODE == "debug":
    print(device)

train_data = ISICDataset(
    image_folder_path="./data/ISIC-2017_Training_Data", 
    mask_folder_path="./data/ISIC-2017_Training_Part1_GroundTruth", 
    diagnoses_path="./data/ISIC-2017_Training_Part3_GroundTruth.csv",
    device=device,
    )
# train_data = torch.utils.data.Subset(train_data, range(200))
train_dataloader = torch.utils.data.DataLoader(
    train_data, 
    batch_size=2, 
    shuffle=True, 
    collate_fn=lambda x:list(zip(*x))
    )

def single_epoch(model, optimizer, dataloader, device, epoch):
    it = 0
    all_losses = []
    for images, targets in tqdm(dataloader):
        # load tensors into GPU
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
        
        # perform backprop
        optimizer.zero_grad()
        losses = model(images, targets)
        losses = sum(loss for loss in losses.values())
        all_losses.append(losses.item())
        print()
        print("Loss", losses.item())
        losses.backward()
        optimizer.step()
        
        it += 1
        # plot the loss every 10th iteration
        if it % 10 == 0:
            plt.plot(list(range(len(all_losses))), all_losses)
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.show()
    
    return all_losses

def train_model(model, dataloader, n_epochs):
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0005, weight_decay=0.0001, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    model.train()
    epoch_losses = {}
    for epoch in tqdm(range(n_epochs)):
        all_losses = single_epoch(model, optimizer, dataloader, device, epoch)
        lr_scheduler.step()
        epoch_losses[epoch] = all_losses
        
    return epoch_losses
        
            
epoch_losses = train_model(model, train_dataloader, 10)
torch.save(model.state_dict(), "Mask_RCNN_ISIC.pt")