import torch
import random
import os
import numpy as np
import PIL.Image as Image
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import glob
from perceiver import Perceiver

"""
Seeding for reproducibility
"""
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

"""
Save best model
"""
def save_best_model(best_valid_loss, model, model_name, epoch, validation_loss):
    if best_valid_loss is None or best_valid_loss > validation_loss:
        torch.save(model, f"{model_name}.pth")
        print(f"MODEL SAVED AT EPOCH {epoch}", flush=True)
        return validation_loss
    return best_valid_loss

"""
Setting Up training arguments based on configuration
"""
def setup_train(config, train_transform, test_transform, data_paths):
    train_dataset = Dataset(data_paths['train_rk'], data_paths['train_lk'], train_transform)
    val_dataset = Dataset(data_paths['val_rk'], data_paths['val_lk'], test_transform)
    test_dataset = Dataset(data_paths['test_rk'], data_paths['test_lk'], test_transform)

    # Tensorboard setup
    writer = SummaryWriter()

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    validation_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    model  = Perceiver(6, 10., 6, input_channels=1, input_axis=2,
            num_latents=512, latent_dim=512, cross_heads=1, latent_heads=8,
            cross_dim_head=64, latent_dim_head=64, num_classes=10, attention_dropout=0.)
    model.to(config['device'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, config['scheduler'])

    return dict(
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        writer=writer
    )

"""
Training Loop
"""
def train_model(training_setup, config):
    # unparse data
    device, train_loader, validation_loader = config['device'], training_setup['train_loader'], training_setup['validation_loader']
    model, optimizer, criterion, scheduler = training_setup['model'], training_setup['optimizer'], training_setup['criterion'], training_setup['scheduler']
    
    writer = training_setup['writer']

    # best validation loss configuration
    best_valid_loss = None

    for epoch in tqdm(range(config['epochs'])):
        # Metric
        total_train_loss = 0
        total_val_loss = 0
        total_train_correct = 0
        total_val_correct = 0

        print("==================================================", flush=True)
        print(f"EPOCH {epoch + 1}", flush=True)

        ### TRAIN MODEL
        model.train()
        for (inputs, labels) in train_loader:
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)
            # forward
            output = model(inputs)
            # calculate loss
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            # update logging
            total_train_loss += loss.item()
            total_train_correct += (output.argmax(dim=1) == labels).sum().item()
        
            if scheduler is not None:
                scheduler.step()
        
        print(f"[TRAIN] EPOCH {epoch + 1} - LOSS: {total_train_loss/len(train_loader)} ACC: {total_train_correct/len(train_loader.dataset)}", flush=True)

        #### EVALUATE MODEL
        model.eval()
        with torch.no_grad():
            for (inputs, labels) in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                # forward
                output = model(inputs)
                # calculate loss
                loss = criterion(output, labels)

                # update logging
                total_val_loss += loss.item()
                total_val_correct += (output.argmax(dim=1) == labels).sum().item()

        print(f"[VAL] EPOCH {epoch + 1} - LOSS: {total_val_loss/len(validation_loader)} ACC: {total_val_correct/len(validation_loader.dataset)}", flush=True)

        # tensorboard logging
        writer.add_scalars('Loss vs Epoch', { 
            "Training Loss": total_train_loss/len(train_loader),
            "Validation Loss": total_val_loss/len(validation_loader)
        }, epoch)

        writer.add_scalars('Accuracy vs Epoch', { 
            "Training Accuracy": total_train_correct/len(train_loader.dataset),
            "Validation Accuracy": total_val_correct/len(validation_loader.dataset)
        }, epoch)

        best_valid_loss = save_best_model(best_valid_loss, model, 
                                          f"perceiver_best_model", epoch + 1, 
                                          total_val_loss/len(validation_loader))
    writer.close()


def model_pipeline(config, train_transform, test_transform, data_paths):
    training_setup = setup_train(config, train_transform, test_transform, data_paths)
    train_model(training_setup, config)

"""
Dataset class, accepts left and right knee array data paths.
"""
class Dataset():
    
    def __init__(self, right_knee, left_knee, transforms):
        self.data = np.concatenate((right_knee, left_knee))

        # right knee - 0
        # left knee - 0
        self.label = torch.cat((torch.zeros(len(right_knee), dtype=torch.long), torch.ones(len(left_knee), dtype=torch.long)))
        self.transforms = transforms
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = Image.open(self.data[idx]).convert('L')
        img = self.transforms(img)
        return img, self.label[idx]


def main():
    seed_everything(42)
    ### PREPROCESS DATASET
    right_knee_data_paths = np.array(sorted(glob.glob("AKOA_Analysis/*RIGHT.nii.gz_*.png")))
    left_knee_data_paths = np.array(sorted(glob.glob("AKOA_Analysis/*LEFT.nii.gz_*.png")))

    ### All these splits have to be done manually using number of images instead of percentage due to slice data issue.
    ### The images are extracted from 3d MRI Scans and split into 40 chunks of 2d images. Hence, to prevent data leakage,
    ### data split process needs to be done manually.

    ### RIGHT KNEE DATA SPLIT
    train_rk, val_rk, test_rk = np.split(right_knee_data_paths, [7920, 7920 + 1320])

    # Height and Width
    height = 228 // 3
    width = 260 // 3

    ### LEFT KNEE DATA SPLIT
    train_lk, val_lk, test_lk = np.split(left_knee_data_paths, [6040, 6040 + 800])

    train_transform = transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ]
    )

    config = dict(
        epochs=20,
        classes=2,
        batch_size=64,
        learning_rate=2e-4,
        scheduler=0.995,
        device="cuda"
    )

    data_paths = dict(
        train_lk=train_lk,
        train_rk=train_rk,
        val_lk=val_lk,
        val_rk=val_rk,
        test_lk=test_lk,
        test_rk=test_rk
    )

    model_pipeline(config, train_transform, test_transform, data_paths)

main()