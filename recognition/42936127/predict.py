import numpy as np
import matplotlib.pyplot as plpt

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import tensorflow as tf

from modules import *
from tools import *
from dataset import *

vqvae = keras.models.load_model("saved_models")
vqvae.compile(optimizer=keras.optimizers.Adam())

predictions = vqvae.predict()