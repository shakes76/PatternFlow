#dataset.py, contains all relevant functions for loading and processing data

# Imports
import cv2
import numpy as np
import os, shutil
import tensorflow as tf
from glob import glob
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from pathlib import Path
from PIL import Image
from os import listdir
from PIL import Image
from matplotlib import pyplot as plt

""" Image Parameters """
H = 256
W = 256
BATCHES = 4

# Loads the isic dataset from the defined path and grabs all the image files
def load_isic(path, split=0.25):
    # Path to the masks being loaded
    masks = (glob(os.path.join(path, "ISIC-2017_Training_Part1_GroundTruth", "*.png")))

    # Path to the images being loaded
    images = (glob(os.path.join(path, "ISIC-2017_Training_Data", "*.jpg")))

    splitAmount = int(split * len(images))

    # Generate a validation set for each image and mask
    trainY, validY = train_test_split(masks, test_size=splitAmount, random_state=42)
    trainX, validX = train_test_split(images, test_size=splitAmount, random_state=42)
    # Generate the final training and test sets
    trainY, testY = train_test_split(trainY, test_size=splitAmount, random_state=42)
    trainX, testX = train_test_split(trainX, test_size=splitAmount, random_state=42)

    trainX, trainY = shuffled(trainX, trainY)
    # Debug output
    print("===Loaded the following data===")
    print(f"Training: {len(trainX)} | {len(trainY)}")
    print(f"Testing: {len(testX)} | {len(testY)}")
    print(f"Validation: {len(validX)} | {len(validY)}")

    # Return the final set
    return (trainX, trainY), (testX, testY), (validX, validY)

# Shuffles the x and y to be randomly ordered
def shuffled(x, y):
    return shuffle(x, y, random_state=42)

# converts a regular data to that of a tensorflow mapped dataset
def tf_dataset_conv(X, Y, batches=4):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    ds = ds.map(tf_parse)
    ds = ds.batch(batches)
    ds = ds.prefetch(batches*2)
    return ds

# TF dataset loading support functions -----------------
def tf_parse(x, y):
    def _parse(x, y):
        return read_data(x, y)
    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])
    return x, y

def read_data(x, y):
    paths = [x.decode(), y.decode()]
    ret = [0, 0]
    for i in range(len(ret)):
        path = paths[i]
        if i == 0:
            x = cv2.imread(path, cv2.IMREAD_COLOR)
        else:
            x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        x = cv2.resize(x, (W, H))
        x = x/255.0
        x = x.astype(np.float32)
        if i == 1:
            x = np.expand_dims(x, axis=-1)
        ret[i] = x
    return ret[0], ret[1]


# ------------------------------------

# Function to allow me to get all the training and test sets I need in one go
def FullLoad(path):
    (trainX, trainY), (testX, testY), (validX, validY) = load_isic(path)

    trainSet = tf_dataset_conv(trainX, trainY, BATCHES)
    testSet = tf_dataset_conv(testX, testY, BATCHES)
    validSet = tf_dataset_conv(validX, validY, BATCHES)

    return trainSet, testSet, validSet


"""
# Data preprocessing
def normalize(input_image, input_mask):
    # Normalize the pixel range values between [0:1]
    img = tf.cast(input_image, dtype=tf.float32) / 255.0
    input_mask -= 1
    return img, input_mask

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

tr, te = load_isic(size=0.05)

plt.imshow(tr[0])
plt.show()
plt.imshow(te[0])
plt.show()
"""
