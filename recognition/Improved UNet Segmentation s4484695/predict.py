import dataset
import modules
import train
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

"""
predict.py script used for loading an already trained model from it's state dictionary.

Inference performed on this trained model with appropriate vizulations.
"""

# PATHS
# Change these values to Relevant Labels, Images folder paths. Images and Labels must be alrady separated.
# IE: Validation, train and test set must already be split into different folders, and also labels and images must be stored in different paths.
testImagesPath = r"PLACEHOLDER, CHANGE THIS TO APPLICABLE PATH"
testLabelsPath = r"PLACEHOLDER, CHANGE THIS TO APPLICABLE PATH"

def main():
    """
    Controls flow of predict.py script when executed as __main__ script. Loads trained model from stat dictionary.
    Loads test data at specified path as specified above.
    Assigns CUDA as device if available.

    return: none
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("Warning CUDA not Found. Using CPU")

    testDataSet = dataset.ISIC2017DataSet(testImagesPath, testLabelsPath, dataset.ISIC_transform_img(), dataset.ISIC_transform_label())
    testDataloader = DataLoader(testDataSet, batch_size=train.batchSize, shuffle=False)

    model = modules.Improved2DUnet()
    model.load_state_dict(torch.load(train.modelPath))
    model.to(device)
    print("Model Successfully Loaded")
    
    test(testDataloader, model, device)

def test(dataLoader, model, device):
    """
    Test inference performed on trained model. Saves vizulations of first 9 image, ground truth and predicted labels. 
    Prints average binary dice loss and dice coefficient over whole test set.

    dataloader: PyTorch DataLoader object
    model: Model of type nn.module
    device: Device being used for training.
    return: none   
    """
    losses_validation = list()
    dice_similarities_validation = list()

    print("> Test Inference Commenced")
    start = time.time()
    model.eval()
    with torch.no_grad():
        for step, (images, labels) in enumerate(dataLoader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            losses_validation.append(train.dice_loss(outputs, labels))
            dice_similarities_validation.append(train.dice_coefficient(outputs, labels))

            if (step == 0):
                train.save_segments(images, labels, outputs, 9, test=True)

        print('Test Loss: {:.5f}, Test Average Dice Similarity: {:.5f}'.format(train.get_average(losses_validation) ,train.get_average(dice_similarities_validation)))
    end = time.time()
    elapsed = end - start
    print("Test Inference took " + str(elapsed/60) + " mins in total")

# If predict.py is executed as main script then train.main()
if __name__ == "__main__":
    main()