import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import tensorflow as tf


import dataset
import modules

modules.model.compile(optimizer='adam', loss='categorical_crossentropy' ,metrics=[modules.dice_similarity])

results = modules.model.fit(dataset.train_generator , steps_per_epoch=dataset.train_steps ,epochs=10,
                              validation_data=dataset.val_generator,validation_steps=dataset.val_steps, verbose=1)
