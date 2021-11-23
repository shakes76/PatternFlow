"""
Functions to load and preprocess data

@author: Jeng-Chung Lien
@student id: 46232050
@email: jengchung.lien@uqconnect.edu.au
"""
import cv2
import glob
import numpy as np
from Modules.misc_utils import progressbar
from sklearn.model_selection import train_test_split
from math import isclose
from time import process_time


def get_min_imageshape(path):
    """
    Function to get the minimum image shape

    Parameters
    ----------
    path : string
      Directory of where the target images are

    Returns
    -------
    min_shape : list
      A list of the minimum image shape [height, width]
    """
    start_time = process_time()
    img_paths = sorted(glob.glob(path))
    length = len(img_paths)
    count = 0
    image_shapes = []

    for img_path in img_paths:
        img = cv2.imread(img_path)
        img_shape = img.shape
        shape_info = [img_shape[0]*img_shape[1], [img_shape[0], img_shape[1]]]
        image_shapes.append(shape_info)
        progressbar(count, length)
        count += 1

    image_shapes = np.array(image_shapes, dtype=object)
    index = np.where(image_shapes[:, 0] == min(image_shapes[:, 0]))
    min_shape = image_shapes[index][0][1]
    print("\nFinished! (Runtime: %s seconds)" % (process_time() - start_time))

    return min_shape


def load_rgbimages(path, height, width):
    """
    Function to load and preprocess rgb images from path to memory

    Parameters
    ----------
    path : string
      Directory of where the target images are
    height : integer
      Parameter to resize the image height
    width : integer
      Parameter to resize the image width

    Returns
    -------
    images : float32 numpy array
      A data type float32 numpy array of the preprocessed images
    """
    start_time = process_time()
    img_paths = sorted(glob.glob(path))
    length = len(img_paths)
    count = 0
    images = []

    for img_path in img_paths:
        # OpenCV reads images as BGR, convert it to RGB when reading
        img = cv2.imread(img_path)
        img = cv2.resize(img, (width, height))
        b, g, r = cv2.split(img)
        img = cv2.merge([r, g, b])
        images.append(img)
        progressbar(count, length)
        count += 1

    images = np.array(images, np.float32)
    images = images / 255.
    print("\nFinished! (Runtime: %s seconds)" % (process_time() - start_time))

    return images


def load_masks(path, height, width):
    """
    Function to load and preprocess masks from path to memory

    Parameters
    ----------
    path : string
      Directory of where the target masks are
    height : integer
      Parameter to resize the mask height
    width : integer
      Parameter to resize the mask width

    Returns
    -------
    images : float32 numpy array
      A data type float32 numpy array of the preprocessed masks
    """
    start_time = process_time()
    mask_paths = sorted(glob.glob(path))
    length = len(mask_paths)
    count = 0
    masks = []

    for mask_path in mask_paths:
        # Read the masks from BGR to grayscale and apply threshold
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = cv2.resize(mask, (width, height))
        mask[mask >= 127] = 255
        mask[mask < 127] = 0
        masks.append(mask)
        progressbar(count, length)
        count += 1

    masks = np.array(masks, np.float32)
    masks = masks / 255.
    masks = masks[:, :, :, np.newaxis]
    print("\nFinished! (Runtime: %s seconds)" % (process_time() - start_time))

    return masks

def train_val_test_split(image_path, mask_path, height, width, split_ratio, randomstate):
    """
    Function to load and preprocess images and mask from path to memory.
    Then split the data according to the split_ratio into train, validation and test sets.

    Parameters
    ----------
    image_path : string
      Directory of where the target masks are
    mask_path : string
      Directory of where the target masks are
    height : integer
      Parameter to resize the image height
    width : integer
      Parameter to resize the image width
    split_ratio : list
      List of the data split condition in the format of [train_ratio, validation_ratio, test_ratio].
      The list must sum up to 1.
    randomstate : integer
      The random seed

    Returns
    -------
    X_train : float32 numpy array
      The train set of data type float32 numpy array of the preprocessed images
    X_val : float32 numpy array
      The validation set of data type float32 numpy array of the preprocessed images
    X_test : float32 numpy array
      The test set of data type float32 numpy array of the preprocessed images
    y_train : float32 numpy array
      The train set of data type float32 numpy array of the preprocessed masks
    y_val : float32 numpy array
      The validation set of data type float32 numpy array of the preprocessed masks
    y_test : float32 numpy array
      The test set of data type float32 numpy array of the preprocessed masks
    """
    # Check if the split ratio is summed up to 1
    if not isclose(sum(split_ratio), 1.):
        raise ValueError("Sum of split_ratio must be 1!")

    # Load the images and masks
    print("\nLoad and preprocess RGB images...")
    images = load_rgbimages(image_path, height, width)
    print("Load and preprocess masks...")
    masks = load_masks(mask_path, height, width)

    # Split the train, validation and test set according to split_ratio
    print("Splitting train set...")
    X_train, X_test, y_train, y_test = train_test_split(images, masks, train_size=split_ratio[0], random_state=randomstate)
    print("Splitting validation and test set...")
    val_split_ratio = split_ratio[1] / (split_ratio[1] + split_ratio[2])
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=val_split_ratio, random_state=randomstate)

    return X_train, X_val, X_test, y_train, y_val, y_test
