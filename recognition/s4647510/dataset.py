import random
from PIL import Image
import numpy as np

batch_size = 32
img_path = "images"

data_train = ""
data_test = ""

def preprocess(img):
    img = np.array(img).astype('float32')
    img = img / 255
    img = img[:, :, np.newaxis]
    return img

data = []
for i in range(batch_size):
    #randomly pick an image from the training datset
    img = random.choice(data_train)
    img = Image.open(img)
    #preprocess the image
    img = preprocess(img)
    data.append(img)
data = np.array(data).astype('float32')

test = []
for i in range(batch_size):
    #randomly pick an image from the training datset
    img = random.choice(data_test)
    img = Image.open(img)
    #preprocess the image
    img = preprocess(img)
    test.append(img)
test = np.array(test).astype('float32')

data_variance = np.var(data / 255.0)
