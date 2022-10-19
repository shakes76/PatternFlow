from cProfile import label
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

# Hyper-parameters
num_epochs = 5
learning_rate = 5 * 10**-2
batchSize = 64
learning_rate_decay = 0.985

validationImagesPath = r"C:\Users\kamra\OneDrive\Desktop\Uni Stuff\2022\COMP3710\Report\Small Data\Validation\Images"
trainImagesPath = r"C:\Users\kamra\OneDrive\Desktop\Uni Stuff\2022\COMP3710\Report\Small Data\Train\Images"
testImagesPath = r"C:\Users\kamra\OneDrive\Desktop\Uni Stuff\2022\COMP3710\Report\Small Data\Test\Images"
validationLabelsPath = r"C:\Users\kamra\OneDrive\Desktop\Uni Stuff\2022\COMP3710\Report\Small Data\Validation\Labels"
trainLabelsPath = r"C:\Users\kamra\OneDrive\Desktop\Uni Stuff\2022\COMP3710\Report\Small Data\Train\Labels"
testLabelsPath = r"C:\Users\kamra\OneDrive\Desktop\Uni Stuff\2022\COMP3710\Report\Small Data\Test\Labels"

# Discovery path only needs to be specified if calling function calculate_mean_std.
discoveryImagesPath = r"C:\Users\kamra\OneDrive\Desktop\Uni Stuff\2022\COMP3710\Report\Data\Train\Images"
discoveryLabelsPath = r"C:\Users\kamra\OneDrive\Desktop\Uni Stuff\2022\COMP3710\Report\Data\Train\Labels"

##################################################################################################################################

# validationImagesPath = r"C:\Users\kamra\OneDrive\Desktop\Uni Stuff\2022\COMP3710\Report\Data\Validation\Images"
# trainImagesPath = r"C:\Users\kamra\OneDrive\Desktop\Uni Stuff\2022\COMP3710\Report\Data\Train\Images"
# testImagesPath = r"C:\Users\kamra\OneDrive\Desktop\Uni Stuff\2022\COMP3710\Report\Data\Test\Images"
# validationLabelsPath = r"C:\Users\kamra\OneDrive\Desktop\Uni Stuff\2022\COMP3710\Report\Data\Validation\Labels"
# trainLabelsPath = r"C:\Users\kamra\OneDrive\Desktop\Uni Stuff\2022\COMP3710\Report\Data\Train\Labels"
# testLabelsPath = r"C:\Users\kamra\OneDrive\Desktop\Uni Stuff\2022\COMP3710\Report\Data\Test\Labels"

# # Discovery path only needs to be specified if calling function calculate_mean_std.
# discoveryImagesPath = r"C:\Users\kamra\OneDrive\Desktop\Uni Stuff\2022\COMP3710\Report\Data\Train\Images"
# discoveryLabelsPath = r"C:\Users\kamra\OneDrive\Desktop\Uni Stuff\2022\COMP3710\Report\Data\Train\Labels"

##################################################################################################################################

# validationImagesPath = "../../../Data/Validation/Images"
# trainImagesPath = "../../../Data/Train/Images"
# testImagesPath = "../../../Data/Test/Images"
# validationLabelsPath = "../../../Data/Validation/Labels"
# trainLabelsPath = "../../../Data/Train/Labels"
# testLabelsPath = "../../../Data/Test/Labels"

# # Discovery path only needs to be specified if calling function calculate_mean_std.
# discoveryImagesPath = trainImagesPath
# discoveryLabelsPath = trainLabelsPath


modelPath = "model.pth"
outputPath = "./Output"

def init():
    validDataSet = dataset.ISIC2017DataSet(validationImagesPath, validationLabelsPath, dataset.ISIC_transform_img(), dataset.ISIC_transform_label())
    validDataloader = DataLoader(validDataSet, batch_size=batchSize, shuffle=False)
    trainDataSet = dataset.ISIC2017DataSet(trainImagesPath, trainLabelsPath, dataset.ISIC_transform_img(), dataset.ISIC_transform_label())
    trainDataloader = DataLoader(trainDataSet, batch_size=batchSize, shuffle=True)
    testDataSet = dataset.ISIC2017DataSet(testImagesPath, testLabelsPath, dataset.ISIC_transform_img(), dataset.ISIC_transform_label())
    testDataloader = DataLoader(testDataSet, batch_size=batchSize, shuffle=False)

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

    train_and_validate(dataLoaders, model, device)

    torch.save(model.state_dict(), modelPath)

def train_and_validate(dataLoaders, model, device):
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

    for epoch in range(num_epochs):
        train_loss, train_coeff = train(dataLoaders["train"], model, device, criterion, optimizer, scheduler)
        valid_loss, valid_coeff = validate(dataLoaders["valid"], model, device, criterion)

        losses_training.append(train_loss)
        dice_similarities_training.append(train_coeff)
        losses_valid.append(valid_loss)
        dice_similarities_valid.append(valid_coeff)


        print ("Epoch [{}/{}], Training Loss: {:.5f}, Training Dice Similarity {:.5f}".format(epoch+1, num_epochs, losses_training[-1], dice_similarities_training[-1]))
        print('Validation Training Loss: {:.5f}, Validation Average Dice Similarity: {:.5f}'.format(get_average(losses_valid) ,get_average(dice_similarities_valid)))
        
    
    end = time.time()
    elapsed = end - start
    print("Training & Validation Took " + str(elapsed/60) + " Minutes")

    save_list_as_scatter(trainList=losses_training, valList=losses_valid, type="Loss", path="LossCurve.png")
    save_list_as_scatter(trainList=dice_similarities_training, valList=dice_similarities_valid, type="Dice Coefficient", path="DiceCurve.png")


def train(dataLoader, model, device, criterion, optimizer, scheduler):

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

def validate(dataLoader, model, device, criterion):

    losses = list()
    coefficients = list()

    model.eval()
    with torch.no_grad():
        for images, labels in dataLoader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            coefficients.append(dice_coefficient(outputs, labels).item())
    
    return get_average(losses), get_average(coefficients)


# Variable numList must be a list of number types only
def get_average(numList):
    size = len(numList)
    count = 0
    for num in numList:
        count += num
    
    return count / size

def dice_loss(outputs, labels):
    return 1 - dice_coefficient(outputs, labels)

# outputs corresponds to u in improved Unet Paper, labels corresponds to v in improved unet paper.
# Note: u must be softmax output of network and v must be a one hot encoding of ground truth segmentations

def dice_coefficient(outputs, labels, epsilon=10**-8):

    # currentBatchSize = len(outputs)
    # smooth = 1.

    # outputs_flat = outputs.view(currentBatchSize, -1)
    # labels_flat = labels.view(currentBatchSize, -1)

    # intersection = (outputs_flat * labels_flat).sum()
    # diceCoefficient = (2. * intersection)/(outputs_flat.sum()+labels_flat.sum()+epsilon)

    # #dims = (0,) + tuple(range(2, labels.ndimension()))
    # intersection = torch.sum(outputs * labels, dims)
    # denom = torch.sum(outputs + labels, dims)
    # diceCoefficient = (2. * intersection / (denom + epsilon)).mean()

    intersection = (outputs * labels).sum()
    denom = (outputs + labels).sum() + epsilon
    diceCoefficient = (2. * intersection) / denom
    return diceCoefficient

def print_model_info(model):
    print("Model No. of Parameters:", sum([param.nelement() for param in model.parameters()]))
    print(model)

def display_test(dataLoader):
    # Display image and label.
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

def save_list_as_scatter(trainList, valList, type, path):
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


main()
