#dataset.py, contains all relevant functions for loading and processing data

# Imports
import numpy as np
import os, shutil
from pathlib import Path
from PIL import Image
from os import listdir
from PIL import Image
from matplotlib import pyplot as plt

# Reads the directory for images of a specified dimension and converts
# To a stacked array of images
# INPUT:
#   Path
#       Filepath for a directory of images
# RETURN:
#   numpy array of images
def read_dir(path, split):
    # Load the path into memory
    images_dir = Path(path).expanduser()

    # Reducing the size of the images so they don't nuke my ram,
    # (I only have 8GB sorry!)
    dim = (128, 128)

    # Load list of data to check
    X_image_train = []
    direct = listdir(images_dir)
    direct = direct[0:int(len(direct)*split)]

    for fname in direct:
        # Generate the file path per file name
        fpath = os.path.join(images_dir, fname)
        im = Image.open(fpath)#.convert('RGB')
        resized = im.resize(dim)
        X_image_train.append(resized)

    # Converting the image to numpy array
    X_image_array=[]
    for x in range(len(X_image_train)):
        X_image=np.array(X_image_train[x],dtype='uint8')
        X_image_array.append(X_image)

    # Stack the array and convert to tensor
    return np.stack(X_image_array)

# Loads the isic dataset
# INPUT:
#   % of data to load, size
#       Default: 1 (100%)
#       Load a percetange of each dataset for quicker testing, scaled 0 - 1
# OUTPUT:
#   training_data
#       numpy array of training images
#   testing_data
#       numpy arrat of testing images
def load_isic(size=1):
    # Declare routes to data to download
    training_data_route = r".\Data\ISIC-2017_Training_Data"
    testing_data_route= r".\Data\ISIC-2017_Test_v2_Data"
    training_truth_route = r".\Data\ISIC-2017_Training_Part1_GroundTruth"
    testing_truth_route = r".\Data\ISIC-2017_Test_v2_Part1_GroundTruth"

    # Load training data from route
    tr_data = read_dir(training_data_route, size)
    print("Training: ", tr_data.shape)

    # Load testing data from route
    te_data = read_dir(testing_data_route, size)
    print("Testing: ", te_data.shape)

    # Load training truth data from route
    trt_data = read_dir(training_truth_route, size)
    print("Training Truth: ", trt_data.shape)

    # Load testing truth data from route
    tet_data = read_dir(testing_truth_route, size)
    print("Testing Truth: ", tet_data.shape)
    return tr_data, te_data, trt_data, tet_data
    
# Test loading data and displaying
"""
tr, te = load_isic(size=0.05)

plt.imshow(tr[0])
plt.show()
plt.imshow(te[0])
plt.show()
"""
