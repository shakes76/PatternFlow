import dataset
import modules
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

# Hyper-parameters
num_epochs = 35
learning_rate = 5 * 10**-4
batchSize = 8
learning_rate_decay = 0.985

def init():
    validDataSet = dataset.ISIC2017DataSet(r"C:\Users\kamra\OneDrive\Desktop\Uni Stuff\2022\COMP3710\Report\Small Data\Validation\Images", r"C:\Users\kamra\OneDrive\Desktop\Uni Stuff\2022\COMP3710\Report\Small Data\Validation\Labels", dataset.ISIC_transform_img(), dataset.ISIC_transform_label())
    validDataloader = DataLoader(validDataSet, batch_size=batchSize, shuffle=True)
    trainDataSet = dataset.ISIC2017DataSet(r"C:\Users\kamra\OneDrive\Desktop\Uni Stuff\2022\COMP3710\Report\Small Data\Train\Images", r"C:\Users\kamra\OneDrive\Desktop\Uni Stuff\2022\COMP3710\Report\Small Data\Train\Labels", dataset.ISIC_transform_img(), dataset.ISIC_transform_label())
    trainDataloader = DataLoader(trainDataSet, batch_size=batchSize, shuffle=True)
    testDataSet = dataset.ISIC2017DataSet(r"C:\Users\kamra\OneDrive\Desktop\Uni Stuff\2022\COMP3710\Report\Small Data\Test\Images", r"C:\Users\kamra\OneDrive\Desktop\Uni Stuff\2022\COMP3710\Report\Small Data\Test\Labels", dataset.ISIC_transform_img(), dataset.ISIC_transform_label())
    testDataloader = DataLoader(testDataSet, batch_size=batchSize, shuffle=True)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("Warning CUDA not Found. Using CPU")

    dataLoaders = dict()
    dataLoaders["valid"] = validDataloader
    dataLoaders["train"] = trainDataloader
    dataLoaders["test"] = testDataloader

    dataSets = dict()
    dataSets["valid"] = validDataSet
    dataSets["train"] = trainDataSet
    dataSets["test"] = testDataSet

    return dataSets, dataLoaders, device

def main():
    dataSets, dataLoaders, device = init()
    model = modules.Improved2DUnet()
    model = model.to(device)

    # Code for Diagnostics/Visualization
    #display_test(dataLoaders["valid"])
    #calculate_mean_std()
    #print_model_info(model)

    # Define optimization parameters and loss according to Improved Unet Paper.
    criterion = dice_loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=10**-5)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=learning_rate_decay)
    totalStep = len(dataLoaders["train"])

    model.train()
    print("Training Commenced:")
    start = time.time()
    for epoch in range(num_epochs):
        for batchStep, (images, labels) in enumerate(dataLoaders["train"]):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batchStep % batchSize == 0:
                print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.5f}"
                    .format(epoch+1, num_epochs, batchStep+1, totalStep, loss.item()))
            
        scheduler.step()
    end = time.time()
    elapsed = end - start
    print("Training Took " + elapsed/60 + " Minutes")

def dice_loss():

    pass

def dice_coefficient():

    pass

def print_model_info(model):
    print("Model No. of Parameters:", sum([param.nelement() for param in model.parameters()]))
    print(model)

def display_test(dataLoader):
    # Display image and label.
    train_features, train_labels = next(iter(dataLoader))
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

def calculate_mean_std():
    psum    = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])
    height = 2016
    width = 3024

    discoveryDataSet = dataset.ISIC2017DataSet(r"C:\Users\kamra\OneDrive\Desktop\Uni Stuff\2022\COMP3710\Report\Data\Train\Images", r"C:\Users\kamra\OneDrive\Desktop\Uni Stuff\2022\COMP3710\Report\Data\Train\Labels", dataset.ISIC_transform_discovery(), dataset.ISIC_transform_label())
    discoveryDataloader = DataLoader(discoveryDataSet, batch_size=batchSize, shuffle=True) 

    display_test(discoveryDataloader)

    for inputs, _ in tqdm(discoveryDataloader):
        psum    += inputs.sum(axis        = [0, 2, 3])
        psum_sq += (inputs ** 2).sum(axis = [0, 2, 3])
    
    # pixel count
    count = discoveryDataSet.ImagesSize * height * width

    # mean and std
    total_mean = psum / count
    total_var  = (psum_sq / count) - (total_mean ** 2)
    total_std  = torch.sqrt(total_var)

    # output
    print('mean: '  + str(total_mean))
    print('std:  '  + str(total_std))


main()
