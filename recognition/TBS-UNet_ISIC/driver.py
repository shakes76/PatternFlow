import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Get filepaths
input_data_dir = "data\\reduced\\ISIC2018_Task1-2_Training_Input_x2"
gt_data_dir = "data\\reduced\\ISIC2018_Task1_Training_GroundTruth_x2"

input_data_paths = os.listdir(input_data_dir)
gt_data_paths = os.listdir(gt_data_dir)

if(len(input_data_paths) != len(gt_data_paths)):
    print("Mismatch in data / labels")

nTrain = len(input_data_paths)

testImg = Image.open(input_data_dir + "\\" + input_data_paths[0])
nx, ny = testImg.size
trainData = np.empty([nTrain, ny, nx, 3])

# iterate over data to construct full input data
for i, path in enumerate(input_data_paths):
    img = Image.open(input_data_dir + "\\" + path).resize((nx, ny))
    trainData[i] = np.asarray(img) / 255.0

nTypes = 2
trainLabels = np.empty([nTrain, ny, nx, nTypes])

# iterate over data to construct full input labels
for i, path in enumerate(gt_data_paths):
    img = Image.open(gt_data_dir + "\\" + path).resize((nx, ny))
    trainLabels[i] = np.eye(nTypes)[(np.asarray(img) * (1/255)).astype(int)]

