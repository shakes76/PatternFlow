import imp
import numpy as np
from tensorflow import keras
import tensorflow_addons as tfa
import module
import dataset

test_ds = dataset.createTestData()
model = module.createModel()

# Loads the weights
model.load_weights('./model.h5')

# Re-evaluate the model
_, acc = model.evaluate(test_ds)
print(f"Test acc: {round(acc * 100, 2)}%")