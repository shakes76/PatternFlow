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
import numpy as np

# Hyper-parameters, can adjust these to affect loss and dice coefficients of training and test of model. 
# These parameters achieve the target goal of >0.8 dice coefficient average on test set.
num_epochs = 30
learning_rate = 5 * 10**-4
batchSize = 16
learning_rate_decay = 0.985

# PATHS
# Change these values to Relevant Labels, Images folder paths. Images and Labels must be alrady separated.
# IE: Validation, train and test set must already be split into different folders, and also labels and images must be stored in different paths.
validationImagesPath = r"PLACEHOLDER, CHANGE THIS TO APPLICABLE PATH"
trainImagesPath = r"PLACEHOLDER, CHANGE THIS TO APPLICABLE PATH"
validationLabelsPath = r"PLACEHOLDER, CHANGE THIS TO APPLICABLE PATH"
trainLabelsPath = r"PLACEHOLDER, CHANGE THIS TO APPLICABLE PATH"
# Discovery path only needs to be specified if calling function calculate_mean_std.
discoveryImagesPath = r"PLACEHOLDER, CHANGE THIS TO APPLICABLE PATH"
discoveryLabelsPath = r"PLACEHOLDER, CHANGE THIS TO APPLICABLE PATH"
# Below is the path to save model to after training is complete
modelPath = r"PLACEHOLDER, CHANGE THIS TO APPLICABLE PATH"


"""
Improved 2D-UNet for Binary Segmentation of ISIC2017 Lesion Data Set.
>0.8 Test Set Accuracy achieved.

Training of U-Net Model takes place in this script, saves model to above specified path for inference or other purposes.

Make sure you have followed instructions above to set paths properly, or unpredictable behaviour for this training algorithm.
"""


def init():
    """
    Initializes Data Sets, checks for CUDA GPU.

    return: dictionary for datasets of validation and training, dictionary for dataloaders of validation and training.
    """
    validDataSet = dataset.ISIC2017DataSet(validationImagesPath, validationLabelsPath, dataset.ISIC_transform_img(), dataset.ISIC_transform_label())
    validDataloader = DataLoader(validDataSet, batch_size=batchSize, shuffle=False)
    trainDataSet = dataset.ISIC2017DataSet(trainImagesPath, trainLabelsPath, dataset.ISIC_transform_img(), dataset.ISIC_transform_label())
    trainDataloader = DataLoader(trainDataSet, batch_size=batchSize, shuffle=True)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("Warning CUDA not Found. Using CPU")

    dataLoaders = dict()
    dataLoaders["valid"] = validDataloader
    dataLoaders["train"] = trainDataloader

    dataSets = dict()
    dataSets["valid"] = validDataSet
    dataSets["train"] = trainDataSet

    return dataSets, dataLoaders, device

def main():
    """
    Controls flow of train.py script when executed as __main__ script. Trains model then saves model upon training completion.

    return: none
    """
    dataSets, dataLoaders, device = init()
    model = modules.Improved2DUnet()
    model = model.to(device)

    # Code for Diagnostics/Visualization & Discovery
    #display_test(dataLoaders["valid"])
    #calculate_mean_std()
    #print_model_info(model)

    train_and_validate(dataLoaders, model, device)

    torch.save(model.state_dict(), modelPath)

def train_and_validate(dataLoaders, model, device):
    """
    Implements training and validation with loss functions, optimizer and scheduler as specified in 
    “Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge,”, with hyper parameters specified above.
    Saves plot of loss and coefficient metrices curve.

    dataloaders: dictionary of PyTorch DataLoader objects
    model: Model of type nn.module
    device: Device being used for training.
    return: none   
    """
    # Define optimization parameters and loss according to Improved Unet Paper.
    criterion = dice_loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=10**-5)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=learning_rate_decay)

    losses_training = list()
    dice_similarities_training = list()
    losses_valid = list()
    dice_similarities_valid = list()

    print("Training and Validation Commenced:")
    start = time.time()
    epochNumber = 0

    for epoch in range(num_epochs):
        epochNumber += 1
        train_loss, train_coeff = train(dataLoaders["train"], model, device, criterion, optimizer, scheduler)
        valid_loss, valid_coeff = validate(dataLoaders["valid"], model, device, criterion, epochNumber)

        losses_training.append(train_loss)
        dice_similarities_training.append(train_coeff)
        losses_valid.append(valid_loss)
        dice_similarities_valid.append(valid_coeff)


        print ("Epoch [{}/{}], Training Loss: {:.5f}, Training Dice Similarity {:.5f}".format(epoch+1, num_epochs, losses_training[-1], dice_similarities_training[-1]))
        print('Validation Loss: {:.5f}, Validation Average Dice Similarity: {:.5f}'.format(get_average(losses_valid) ,get_average(dice_similarities_valid)))
        
    
    end = time.time()
    elapsed = end - start
    print("Training & Validation Took " + str(elapsed/60) + " Minutes")

    save_list_as_plot(trainList=losses_training, valList=losses_valid, type="Loss", path="LossCurve.png")
    save_list_as_plot(trainList=dice_similarities_training, valList=dice_similarities_valid, type="Dice Coefficient", path="DiceCurve.png")


def train(dataLoader, model, device, criterion, optimizer, scheduler):
    """
    Completes one epoch of training.

    dataloader: PyTorch DataLoader object
    model: Model of type nn.Module
    device: Device being used for training.
    criterion: function returning a function for calculating loss.
    optimizer: torch.optim object
    scheduler: torch.optim.scheduler object
    return: average loss and dice coefficient once completed current epoch   
    """

    model.train()

    losses = list()
    coefficients = list()

    for images, labels in dataLoader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        losses.append(loss.item())
        coefficients.append(dice_coefficient(outputs, labels).item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    scheduler.step()

    return get_average(losses), get_average(coefficients)

def validate(dataLoader, model, device, criterion, epochNumber):
    
    """
    Completes one epoch of training.

    dataloader: PyTorch DataLoader object
    model: Model of type nn.Module
    device: Device being used for training.
    criterion: function returning a function for calculating loss.
    epochNumber: current epoch
    return: average loss and dice coefficient once completed current epoch   
    """

    losses = list()
    coefficients = list()

    model.eval()
    with torch.no_grad():
        for step, (images, labels) in enumerate(dataLoader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            coefficients.append(dice_coefficient(outputs, labels).item())

            if (step == 0):
                save_segments(images, labels, outputs, 9, epochNumber)
    
    return get_average(losses), get_average(coefficients)


# Variable numList must be a list of number types only
def get_average(numList):
    """
    Calculates Averages of a list of number types.

    numList: a List object containing only number types
    return: number type (float, int, etc...)
    """
    size = len(numList)
    count = 0
    for num in numList:
        count += num
    
    return count / size

def dice_loss(outputs, labels):
    """
    Calculates binary dice loss given predicted outputs and ground truth labels.

    outputs: torch array of predicted outputs
    labels: torch array of ground truth labels
    return: binary dice loss
    """
    return 1 - dice_coefficient(outputs, labels)

def dice_coefficient(outputs, labels, epsilon=10**-8):
    """
    Calculates binary dice coefficient given predicted outputs and ground truth labels.

    outputs: torch array of predicted outputs
    labels: torch array of ground truth labels
    epsilon: small value to prevent 0 division error, this value should not be changed unless you have reasons to.
    return: binary dice coefficient
    """

    intersection = (outputs * labels).sum()
    denom = (outputs + labels).sum() + epsilon
    diceCoefficient = (2. * intersection) / denom
    return diceCoefficient

def print_model_info(model):
    """
    Provides information about nn.Module object.

    model: nn.Module
    """
    print("Model No. of Parameters:", sum([param.nelement() for param in model.parameters()]))
    print(model)

def display_test(dataLoader):
    """
    Functions for testing dataLoader data has been loaded as expected. Helpful for vizualising loaded data.

    dataloader: PyTorch DataLoader object
    """

    train_features, train_labels = next(iter(dataLoader))
    currentBatchSize = train_features.size()[0]
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")

    _, axs = plt.subplots(currentBatchSize, 2)
    for row in range(currentBatchSize):
        img = train_features[row]
        img = img.permute(1,2,0).numpy()
        label = train_labels[row]
        label = label.permute(1,2,0).numpy()
        axs[row][0].imshow(img)
        axs[row][1].imshow(label, cmap="gray")

    plt.show()

def save_segments(images, labels, outputs, numComparisons, epochNumber=num_epochs, test=False):
    """
    Saves the first numComparisons images with corresponding ground truth labels and predicted labels for easy and succinct visualization.

    images: torch array of images.
    outputs: torch array of predicted outputs.
    labels: torch array of ground truth labels.
    numComparisons: number of image, ground truth and predictions to save for comparison.
    epochNumber: epoch number of corresponding segments.
    test: True if saving segments for test set, false for training or validation.
    """

    if numComparisons > batchSize:
        numComparisons = batchSize
    
    images=images.cpu()
    labels=labels.cpu()
    outputs=outputs.cpu()

    fig, axs = plt.subplots(numComparisons, 3)
    axs[0][0].set_title("Image")
    axs[0][1].set_title("Ground Truth")
    axs[0][2].set_title("Predicted")
    for row in range(numComparisons):
        img = images[row]
        img = img.permute(1,2,0).numpy()
        label = labels[row]
        label = label.permute(1,2,0).numpy()
        pred = outputs[row]
        pred = pred.permute(1,2,0).numpy()
        axs[row][0].imshow(img)
        axs[row][0].xaxis.set_visible(False)
        axs[row][0].yaxis.set_visible(False)

        axs[row][1].imshow(label, cmap="gray")
        axs[row][1].xaxis.set_visible(False)
        axs[row][1].yaxis.set_visible(False)

        axs[row][2].imshow(pred, cmap="gray")
        axs[row][2].xaxis.set_visible(False)
        axs[row][2].yaxis.set_visible(False)
    
    if (not test):
        fig.suptitle("Validation Segments Epoch: " + str(epochNumber))
        #fig.tight_layout()
        plt.savefig("ValidationSegmentsEpoch" + str(epochNumber))
    else:
        fig.suptitle("Test Segments")
        #fig.tight_layout()
        plt.savefig("TestSegments")
    plt.close()

def save_list_as_plot(trainList, valList, type, path):
    """
    Saves a plot of two lists of type to path, each entry pair corresponds to an epoch.

    trainList: list of number type and same size as valList
    valList: list of number type and same size as trainList
    type: the type of data stored in the lists
    path: path to save plot to
    """
    if (len(trainList) != len(valList)):
        print("ERROR: Cannot display!")
    
    length = len(trainList)
    xList = list()
    x = 1
    for i in range(length):
        xList.append(x)
        x += 1

    plt.xticks(np.arange(min(xList), max(xList)+1, 1.0))
    plt.plot(xList, trainList, label="Training " + type)
    plt.plot(xList, valList, label="Validation " + type)
    plt.legend()
    plt.title("Training and Validation " + type + " Over Epochs")
    plt.savefig(fname=path)
    plt.close()

def calculate_mean_std():
    """
    Used for discovering mean and standard deviation for normalization of ISIC2017 DataSet.
    """
    psum    = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])
    height = 2016
    width = 3024

    discoveryDataSet = dataset.ISIC2017DataSet(discoveryImagesPath, discoveryLabelsPath, dataset.ISIC_transform_discovery(), dataset.ISIC_transform_label())
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

# If train.py is executed as main script then train.main()
if __name__ == "__main__":
    main()
