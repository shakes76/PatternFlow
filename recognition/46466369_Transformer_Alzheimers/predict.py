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

def load_weights():
    train.vit_model.load_weights(train.file_path)


def predict_single_image(image_filepath):
    image = np.array([np.array(Image.open(image_filepath).resize((train.IMAGE_SIZE, train.IMAGE_SIZE)))])
    load_weights()
    train.vit_model.predict(image)