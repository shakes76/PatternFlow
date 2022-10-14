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
    batch_size = 32
    img_path = "images"

    data_train = []
    train_path = os.path.join(img_path, "/train")
    for filename in os.listdir(train_path):
        image_id = filename[5:]
        data_train.append(os.path.join(train_path, filename))

    data_test = []
    test_path = os.path.join(img_path, "/test")
    for filename in os.listdir(test_path):
        image_id = filename[5:]
        data_train.append(os.path.join(test_path, filename))
    
    train = []
    for i in range(batch_size):
        img = random.choice(data_train)
        img = Image.open(img)
        img = preprocess(img)
        train.append(img)
        train = np.array(train).astype('float32')

    test = []
    for i in range(batch_size):
        img = random.choice(data_test)
        img = Image.open(img)
        img = preprocess(img)
        test.append(img)
        test = np.array(test).astype('float32')
    
    data_variance = np.var(train / 255.0)
    
    return (train, test)

