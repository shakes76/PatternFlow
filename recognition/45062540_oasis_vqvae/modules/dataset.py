import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import os

def load_data():
    """
    Store the filenames of the images from the specified directory. 
    
    Params: None
        
    Returns:
        Three lists containing filnames of all the images in the train, validate and test dataset respectively.
    """

    data_train = []
    for filename in os.listdir("D:/keras_png_slices_data/keras_png_slices_train"):
        image_id = filename[5:]
        data_train.append(os.path.join("D:/keras_png_slices_data/keras_png_slices_train", filename))

    #store the filenames of the validation dataset
    data_validate = []
    for filename in os.listdir("D:/keras_png_slices_data/keras_png_slices_validate"):
        image_id = filename[5:]
        data_validate.append(os.path.join("D:/keras_png_slices_data/keras_png_slices_validate", filename))

    #store the filenames of the testing dataset
    data_test = []
    for filename in os.listdir("D:/keras_png_slices_data/keras_png_slices_test"):
        image_id = filename[5:]
        data_test.append(os.path.join("D:/keras_png_slices_data/keras_png_slices_test", filename))
    
    return data_train, data_validate, data_test

def preprocess_image(img):
    """
    Preprocess the image into a specific format before giving to the model to train
    
    Params:
        img: the image to preprocess
        
    Returns:
        the image array after preoprocessing
    """
    img = np.array(img).astype('float32')
    #scale image pixels in (0,1)
    img = img / 255
    img = img[:, :, np.newaxis]
    return img

def data_generator(train_data, batch_size = 8):
    """
    A generator randomly selects batch_size number of images from the training dataset
    
    Params:
        batch_size: the number of images to be selected from the training dataset, default value = 8
        train_data: a list containing filenames of all the images in the training dataset.
        
    Returns:
        a list of batch_size number of images randomly selected from the training dataset
    """
    while True:
        xs = []
        for i in range(batch_size):
            img = random.choice(train_data)
            img = Image.open(img)
            img = preprocess_image(img)
            xs.append(img)
        xs = np.array(xs).astype('float32')
        yield xs

def validate_generator(validate_data, batch_size = 8):
    """
    A generator outputs batch_size number of images from the validate dataset in sequential order
    
    Params:
        batch_size: the number of images to be selected from the validate dataset, default value = 8
        validate_data: a list containing filenames of all the images in the validate dataset.
        
    Returns:
        a list of batch_size number of images selected from the validate dataset
    """
    count = 0
    while True:
        xs = []
        for i in range(batch_size):
            if count == len(validate_data):
                count = 0
            img = validate_data[count]
            count += 1
            img = Image.open(img)
            img = preprocess_image(img)
            xs.append(img)
        xs = np.array(xs).astype('float32')
        yield xs
