from math import floor
import os
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras.engine import training
import numpy as np
import matplotlib.pyplot as plt

from model import UNet
from trianing_utils import DataGenerator

# Generate a test UNet model to check if it compiles correctly
model = UNet()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
model.summary()
plot_model(model, show_shapes=True)

img_size = 128
img_channels = 3
batch_size = 20

base_path = "./data/ISIC2018_Task1-2_Training_Data/"
image_path = "./dataset/ISIC2018_Task1-2_Training_Data/images"

training_ids = []
for file in os.walk(image_path):
    for filename in file:
        training_ids.append(filename)

# Gets filenames and takes .jpg off to leave IDs
training_ids = training_ids[2]
training_ids = [filename.split('.',1)[0] for filename in training_ids]

# Split into training, test and validation data
validation_size = floor(len(training_ids) * 0.2)
test_size = floor(validation_size * 0.5)

# Split data into test, train validation
validation_test_ids = training_ids[:validation_size]
test_ids = validation_test_ids[:test_size]
validation_ids = validation_test_ids[test_size]
training_ids = training_ids[validation_size:]

print("Total images: = ", len(test_ids + training_ids + validation_ids))
print("Training images: ", len(training_ids))
print("Testing images: ", len(test_ids))
print("Validation images: ", len(validation_ids))


# Instantiate a Generator
gen = DataGenerator(training_ids, base_path, batch_size=batch_size, img_size=img_size)

# Test the Generator
testX, testY = gen.__getitem__(0)
print("First image: ", testX.shape, testY.shape)

plot = plt.figure()
plot.subplots_adjust(hspace=0.5, wspace=0.5)
ax = plot.add_subplot(1, 2, 1)
ax.imshow(testX[0])
ax = plot.add_subplot(1, 2, 2)
ax.imshow(np.reshape(testY[0], (img_size, img_size)), cmap="gray")