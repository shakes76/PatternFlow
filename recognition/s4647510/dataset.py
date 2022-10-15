import random
from PIL import Image
import numpy as np
import tensorflow as tf
import os

def preprocess(img):
    img = np.array(img).astype('float32')
    img = img / 255
    img = img[:, :, np.newaxis]
    return img

def load_data():
    img_path = "images/"

    data_train = []
    train_path = os.path.join(img_path, "train/")
    for filename in os.listdir(train_path):
        data_train.append(os.path.join(train_path, filename))

    data_test = []
    test_path = os.path.join(img_path, "test/")
    for filename in os.listdir(test_path):
        data_test.append(os.path.join(test_path, filename))
        
    data_validate = []
    validate_path = os.path.join(img_path, "validate/")
    for filename in os.listdir(validate_path):
        data_validate.append(os.path.join(validate_path, filename))
    
    train = []
    for img in data_train:
        img = Image.open(img)
        img = preprocess(img)
        train.append(img)
    train = np.array(train).astype('float32')

    test = []
    for img in data_test:
        img = Image.open(img)
        img = preprocess(img)
        test.append(img)
    test = np.array(test).astype('float32')

    validate = []
    for img in data_validate:
        img = Image.open(img)
        img = preprocess(img)
        validate.append(img)
    validate = np.array(validate).astype('float32')
        
    return (train, test, validate)
