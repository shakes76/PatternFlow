import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import tensorflow as tf

batch_size=128
image_size=64
channels=3
num_epochs=20
learn_rate=0.0002
beta1=0.5