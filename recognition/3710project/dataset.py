
import os
import cv2 as cv
import SimpleITK as sitk
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import skimage
from skimage import io

# ISIC data format using 2017 ISIC data

# Because original data from ISIC has multiple different size, so we 
# need to reshape them into a better size for doing ML
# if data_reshape is True, then the program will create a new data folder in
# the given direction and reshape all the ISIC images into given size
# (.csv will still in the original position and will not change)

def create_data(data_from, data_images, data_to, img_size = 256):
    """ Create data image based on the given data image to the given direction.
        new data should have the given image size.

    Args:
        data_from (String): From direction
        data_images (list): list of images
        data_to (String): To direction
        img_size (int, optional): image size. Defaults to 256.
    """
    for i in data_images:     
        img=sitk.ReadImage(os.path.join(data_from,i))
        img_array=sitk.GetArrayFromImage(img)
        new_array=cv.resize(img_array,(img_size,img_size))
        data_name = i[:-4] # removing last four (.jpg/.png/...)

        io.imsave(data_to + data_name + '.png', new_array)


def reshape_data():
    """ 
    Reshape image into given size and save into the new file
    """
    # training image path
    train_path = './data/data_ISIC/ISIC-2017_Training_Data/'
    train = [fn for fn in os.listdir(train_path) if fn.endswith('jpg')]
    train.sort()

    # training ground truth path
    train_path_gt = './data/data_ISIC/ISIC-2017_Training_Part1_GroundTruth/'
    train_gt = [fn for fn in os.listdir(train_path_gt) if fn.endswith('png')]
    train_gt.sort()

    # test image path
    test_path = './data/data_ISIC/ISIC-2017_Test_v2_Data'
    test = [fn for fn in os.listdir(test_path) if fn.endswith('jpg')]
    test.sort()

    # test ground truth images
    test_path_gt ='./data/data_ISIC/ISIC-2017_Test_v2_Part1_GroundTruth'
    test_gt = [fn for fn in os.listdir(test_path_gt) if fn.endswith('png')]
    test_gt.sort()

    # validation image path
    val_path = './data/data_ISIC/ISIC-2017_Validation_Data'
    val = [fn for fn in os.listdir(val_path) if fn.endswith('jpg')]
    val.sort()

    # validation image path
    val_path_gt = './data/data_ISIC/ISIC-2017_Validation_Part1_GroundTruth'
    val_gt = [fn for fn in os.listdir(val_path_gt) if fn.endswith('png')]
    val_gt.sort()

    if not os.path.exists('./data/data_Reshaped'):
        os.mkdir('./data/data_Reshaped/')
        os.mkdir('./data/data_Reshaped/Train')
        os.mkdir('./data/data_Reshaped/Train_GT')
        os.mkdir('./data/data_Reshaped/Test')
        os.mkdir('./data/data_Reshaped/Test_GT')
        os.mkdir('./data/data_Reshaped/Val')
        os.mkdir('./data/data_Reshaped/Val_GT')

    create_data(train_path, train, './data/data_Reshaped/Train/')
    create_data(train_path_gt, train_gt, './data/data_Reshaped/Train_GT/')
    create_data(test_path, test, './data/data_Reshaped/Test/')
    create_data(test_path_gt, test_gt, './data/data_Reshaped/Test_GT/')
    create_data(val_path, val, './data/data_Reshaped/Val/')
    create_data(val_path_gt, val_gt, './data/data_Reshaped/Val_GT/')


def load_data(csv, path_image, path_image_gt):
    """ load the data and its mask from the given path with the order in csv file
        csv file only use for getting the names of the images

    Args:
        csv (pandas.core.frame.DataFrame): csv file by using pandas to read
        path_image (String): image path 
        path_image_gt (String): image ground truth path

    Returns:
        numpy.ndarray, numpy.ndarray: return numpy.ndarray of images and it's mask
    """
    x, y = [], []
    for _, i in csv.iterrows():
        image = sitk.ReadImage(path_image + i[0]+'.png')
        image_array_ = sitk.GetArrayFromImage(image)
        image_array = image_array_/255.0
        x.append(image_array)
        
        mask_ = cv.imread(path_image_gt + i[0]+'_segmentation.png')
        mask = mask_/255.0
        y.append(mask)
        
    return np.array(x), np.array(y)


def load_dataset(data_reshape = False):
    """ Load the dataset, if the data need to reshape(data_reshape = True) or if there is no 
        reshaped file, then reshape the dataset

    Args:
        data_reshape (bool, optional): reshape the data if True. Defaults to False.

    Returns:
        numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray:
            return the image and its mask image for all training and testing data 
    """
    if data_reshape or not os.path.exists('./data/data_Reshaped'): 
        reshape_data()

    train_csv = pd.read_csv('./data/data_ISIC/ISIC-2017_Training_Data/ISIC-2017_Training_Data_metadata.csv')
    test_csv = pd.read_csv('./data/data_ISIC/ISIC-2017_Test_v2_Data/ISIC-2017_Test_v2_Data_metadata.csv')

    path_train = './data/data_Reshaped/Train/'
    path_train_gt = './data/data_Reshaped/Train_GT/'

    path_test = './data/data_Reshaped/Test/'
    path_test_gt = './data/data_Reshaped/Test_GT/'

    train_x, train_y = load_data(train_csv, path_train, path_train_gt)
    test_x, test_y = load_data(test_csv, path_test, path_test_gt)

    return train_x, train_y, test_x, test_y




