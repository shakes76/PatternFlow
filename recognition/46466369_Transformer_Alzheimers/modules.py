#import tensorflow as tf
import numpy as np
#from tensorflow import keras
#from keras import layers
#import tensorflow_addons as tfa
import dataset
import train
import predict

train_data, test_data = dataset.load_dataset()