from dataset import ISICDataset, get_transform
from modules import get_model
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import pickle
import gc

gc.collect()
torch.cuda.empty_cache()

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
    transform=get_transform(True),
    )
train_data = torch.utils.data.Subset(train_data, range(250))
train_dataloader = torch.utils.data.DataLoader(
    train_data, 
    batch_size=2, 
    shuffle=True, 
    collate_fn=lambda x:tuple(zip(*x))
    )

def single_epoch(model, optimizer, dataloader, device, epoch):
    it = 0
    classificiation_loss = []
    box_regression_loss = []
    mask_loss = []
    total_losses = []
    for images, targets in tqdm(dataloader):
        # load tensors into GPU
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
        
        # perform backprop
        optimizer.zero_grad()
        losses = model(images, targets)
        total_loss = sum(loss for loss in losses.values())
        classificiation_loss.append(losses["loss_classifier"].item())
        box_regression_loss.append(losses["loss_box_reg"].item())
        mask_loss.append(losses["loss_mask"].item())
        total_losses.append(total_loss)
        total_loss.backward()
        optimizer.step()
        it += 1
        if it % 10 == 0:
            plt.plot(classificiation_loss)
            plt.title("Loss Classifier")
            plt.show()
            
            plt.plot(box_regression_loss)
            plt.title("Loss Box Regression")
            plt.show()
            
            plt.plot(mask_loss)
            plt.title("Loss Mask")
            plt.show()
    
    return total_losses

def train_model(model, dataloader, n_epochs):
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0025, weight_decay=0.0001, momentum=0.9)
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

torch.save(model.state_dict(), "Mask_RCNN_ISIC.pt")     

# Save loss dictionary to perform visualisations
epoch_losses = train_model(model, train_dataloader, 1)
with open('epoch_losses.pickle', 'wb') as handle:
    pickle.dump(epoch_losses, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
