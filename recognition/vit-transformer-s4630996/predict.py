import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from modules import vit_classifier
from dataset import import_data
import matplotlib.pyplot as plt
import numpy as np
from config import *
import random
import os
from PIL import Image


# instantiate model
vit_classifier = vit_classifier()

# select optimzer
optimizer = tfa.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# compile the model
vit_classifier.compile(
    optimizer=optimizer,
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
)

# create checkpoint callback
parent_directory = os.getcwd()
checkpoint_filepath = os.path.join(parent_directory, "checkpoint")

# draw a sample from test data
test_NC_filenames = os.listdir(r"AD_NC_square\test\AD")
x_sample_file = random.sample(test_NC_filenames, 1)[0]
x_sample_path = r"AD_NC_square\test\AD\{}".format(x_sample_file)

# open and display the image
x_image = Image.open(x_sample_path)
plt.imshow(x_image)
plt.show()

# convert the image to a numpy array
x_data = np.asarray(x_image)
x_data = np.reshape(x_data, newshape=(-1, 240, 240, 1))

# load the model weights
parent_directory = os.getcwd()
checkpoint_filepath = os.path.join(parent_directory, "checkpoint", "checkpoint.hdf5")
vit_classifier.load_weights(checkpoint_filepath)

# obtain prediction
prediction = np.argmax(vit_classifier.predict(x_data)[0])

# if index 0 prediction is Alzheimers and if index is 1 prediction is normal cognition
if prediction == 0:
    print("Prediction is Alzheimer's Disease for this sample.")
else:
    print("Prediction is normal cognition for this sample.")