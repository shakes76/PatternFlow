from dataset import ISICDataset, get_transform
from modules import get_model
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
import pickle

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
train_data = torch.utils.data.Subset(train_data, range(100))
train_dataloader = torch.utils.data.DataLoader(
    train_data, 
    batch_size=2, 
    shuffle=True, 
    collate_fn=lambda x:tuple(zip(*x)),
    # num_workers=4
    )

test_data = ISICDataset(
    image_folder_path="./data/ISIC-2017_Test_Data", 
    mask_folder_path="./data/ISIC-2017_Test_v2_Part1_GroundTruth", 
    diagnoses_path="./data/ISIC-2017_Test_v2_Part3_GroundTruth.csv",
    device=device,
    transform=get_transform(True),
    )
# train_data = torch.utils.data.Subset(train_data, range(250))
test_dataloader = torch.utils.data.DataLoader(
    train_data, 
    batch_size=2, 
    shuffle=True, 
    collate_fn=lambda x:tuple(zip(*x)),
    # num_workers=4
    )

def single_epoch(model, optimizer, dataloader, device, epoch):
    it = 0
    total_losses = []
    for images, targets in tqdm(dataloader):
        # load tensors into GPU
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
        
        # perform backprop
        optimizer.zero_grad()
        losses = model(images, targets)
        total_loss = sum(loss for loss in losses.values())
        total_loss.backward()
        # nn.utils.clip_grad_value_(model.parameters(), 5)
        optimizer.step()
        total_losses.append(total_loss.detach().cpu().item())
        
        it += 1
        if it % 10 == 0:            
            fig, ax = plt.subplots()
            ax.plot(total_losses)
    
    return total_losses

def train_model(model, train_dataloader, test_dataloader, n_epochs):
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0025, weight_decay=0.0001, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    model.train()
    training_loss = []
    testing_loss = []
    for epoch in tqdm(range(n_epochs)):
        all_losses = single_epoch(model, optimizer, train_dataloader, device, epoch)
        lr_scheduler.step()
        training_loss.append(sum(all_losses))
        with torch.no_grad():
            all_losses = []
            for images, targets in tqdm(test_dataloader):
                images = [image.to(device) for image in images]
                targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
                losses = model(images, targets)
                total_loss = sum(loss for loss in losses.values())
                all_losses.append(total_loss.detach().cpu().item())
                
            testing_loss.append(sum(all_losses))
                
        fig, ax = plt.subplots()
        ax.plot(training_loss, label="Train")
        ax.plot(testing_loss, label="Test")
        ax.legend()
                
        
        
    return training_loss, testing_loss

if __name__ == "__main__":
    training_loss, testing_loss = train_model(model, train_dataloader, test_dataloader, 20)
    torch.save(model.state_dict(), "Mask_RCNN_ISIC4.pt")
    with open('training_loss.pickle', 'wb') as handle:
        pickle.dump(training_loss, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    with open('testing_loss.pickle', 'wb') as handle:
        pickle.dump(testing_loss, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
