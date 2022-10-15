import random
from PIL import Image
import numpy as np
import os

def preprocess(img):
    img = np.array(img).astype('float32')
    img = img / 255
    img = img[:, :, np.newaxis]
    return img

def load_data():
    batch_size = 8
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
    imgs = random.sample(data_train, batch_size)
    for img in imgs:
        img = Image.open(img)
        img = preprocess(img)
        train.append(img)
    train = np.array(train).astype('float32')

    test = []
    imgs = random.sample(data_test, batch_size)
    for img in imgs:
        img = Image.open(img)
        img = preprocess(img)
        test.append(img)
    test = np.array(test).astype('float32')

    validate = []
    imgs = random.sample(data_validate, batch_size)
    for img in imgs:
        img = Image.open(img)
        img = preprocess(img)
        validate.append(img)
    validate = np.array(validate).astype('float32')
        
    return (train, test, validate)
