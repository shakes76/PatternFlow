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

    # Code for Diagnostics/Visualization & Discovery
    #display_test(dataLoaders["valid"])
    #calculate_mean_std()
    #print_model_info(model)

    train(dataLoaders, model, device)
    validate(dataLoaders, model, device)

def train(dataLoaders, model, device):
    # Define optimization parameters and loss according to Improved Unet Paper.
    criterion = dice_loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=10**-5)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=learning_rate_decay)
    totalStep = len(dataLoaders["train"])

    losses_training = list()
    dice_similarities_training = list()

    model.train()
    print("Training and Validation Commenced:")
    start = time.time()
    for epoch in range(num_epochs):
        for batchStep, (images, labels) in enumerate(dataLoaders["train"]):
            images = images.to(device)
            labels = labels.to(device)

            outputs, seg_layers = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batchStep % 10 == 0:
                losses_training.append(loss.item())
                dice_similarities_training.append(dice_coefficient(seg_layers, labels))
                print ("Epoch [{}/{}], Step [{}/{}], Training Loss: {:.5f}, Training Dice Similarity {:.5f}"
                    .format(epoch+1, num_epochs, batchStep+1, totalStep, losses_training[-1], dice_similarities_training[-1]))
            
        scheduler.step()
    end = time.time()
    elapsed = end - start
    print("Training Took " + elapsed/60 + " Minutes")

def validate(dataLoaders, model, device):
    losses_validation = list()
    dice_similarities_validation = list()

    print("> Validation")
    start = time.time()
    model.eval()
    with torch.no_grad():
        for images, labels in dataLoaders["valid"]:
            images = images.to(device)
            labels = labels.to(device)

            outputs, seg_layers = model(images)
            losses_validation.append(dice_loss(seg_layers, labels))
            dice_similarities_validation.append(dice_coefficient(seg_layers, labels))

        print('Validation Training Loss: {:.5f}, Validation Average Dice Similarity: {:.5f}'.format(get_average(losses_validation) ,get_average(dice_similarities_validation)))
    end = time.time()
    elapsed = end - start
    print("Validation took " + elapsed/60 + " mins in total") 


# Variable numList must be a list of number types only
def get_average(numList):
    size = len(numList)
    count = 0
    for num in numList:
        count += num
    
    return count / size

def dice_loss(outputs, labels):
    return 1 - dice_coefficient(outputs, labels)

def dice_coefficient(outputs, labels, epsilon=10**-8):
    K = outputs.shape[1]
    if K == 1:
        labels_1_hot = torch.eye(K + 1)[labels.squeeze(1)]
        labels_1_hot = labels_1_hot.permute(0, 3, 1, 2).float()
        labels_1_hot_f = labels_1_hot[:, 0:1, :, :]
        labels_1_hot_s = labels_1_hot[:, 1:2, :, :]
        labels_1_hot = torch.cat([labels_1_hot_s, labels_1_hot_f], dim=1)
        prob = torch.sigmoid(outputs)
        min_prob = 1 - prob
        probs = torch.cat([prob, min_prob], dim=1)
    else:
        labels_1_hot = torch.eye(K)[labels.squeeze(1)]
        labels_1_hot = labels_1_hot.permute(0, 3, 1, 2).float()
        probs = nn.Softmax(outputs, dim=1)
    
    labels_1_hot = labels_1_hot.type(outputs.type())
    dims = (0,) + tuple(range(2, labels.ndimension()))
    intersection = torch.sum(probs * labels_1_hot, dims)
    denom = torch.sum(probs + labels_1_hot, dims)
    dice_coefficient = (2. * intersection / (denom + epsilon)).mean()
    return dice_coefficient

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
