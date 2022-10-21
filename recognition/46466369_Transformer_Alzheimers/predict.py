import train
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow import keras
from keras import layers
import dataset
import modules
import matplotlib.pyplot as plt
import train


def predict_single_image(image):
    image = np.array([np.array(Image.open(image).resize((train.IMAGE_SIZE, train.IMAGE_SIZE)))])
    train.vit_model.predict(image)