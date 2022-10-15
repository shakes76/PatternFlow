"""
“train.py" containing the source code for training, validating, testing and saving your model. The model
should be imported from “modules.py” and the data loader should be imported from “dataset.py”. Make
sure to plot the losses and metrics during training
"""

import dataset as data
import modules
import tensorflow as tf
import numpy as np
from matplotlib import pyplot
from matplotlib import image

# Download Data and then unzip
#download_oasis()

""" PROCESS TRAINING DATA"""
# Load the training data from the Oasis Data set
train_X = data.load_training ("C:/Users/dapmi/OneDrive/Desktop/Data/oa-sis.tar/keras_png_slices_data/keras_png_slices_train")

# Check Images
pyplot.imshow(train_X[2])
pyplot.show()

# Pre process training data set
train_X = data.process_training(train_X)

# Load the validaton data from the oasis Data set 
validate_X = data.load_training ("C:/Users/dapmi/OneDrive/Desktop/Data/oa-sis.tar/keras_png_slices_data/keras_png_slices_validate")

# Check Images
pyplot.imshow(validate_X[2])
pyplot.show()

# Pre process validation data set
validate_X = data.process_training(validate_X)


# Load the test data from the oasis Data Set 
test_X = data.load_training ("C:/Users/dapmi/OneDrive/Desktop/Data/oa-sis.tar/keras_png_slices_data/keras_png_slices_test")

# Check Images
pyplot.imshow(test_X[2])
pyplot.show()

# Pre process test data set
test_X = data.process_training(test_X)

""" PROCESS TRAINING LABELS DATA """
# Load the segmented training labels data from the Oasis Data set
train_Y = data.load_labels ("C:/Users/dapmi/OneDrive/Desktop/Data/oa-sis.tar/keras_png_slices_data/keras_png_slices_seg_train")
# Pre process training labels data
train_Y = data.process_labels(train_Y)

# Check Images
pyplot.imshow(train_Y[2,:,:,3])
pyplot.show()

# Load the segmented validation labels data from the Oasis Data set
validate_Y = data.load_labels("C:/Users/dapmi/OneDrive/Desktop/Data/oa-sis.tar/keras_png_slices_data/keras_png_slices_seg_validate")
# Pre process validation labels data
validate_Y = data.process_labels(validate_Y)
 
# Check Images
pyplot.imshow(validate_Y[2,:,:,3])
pyplot.show()

# Load the segmented test labels data from the Oasis Data set
test_Y = data.load_labels("C:/Users/dapmi/OneDrive/Desktop/Data/oa-sis.tar/keras_png_slices_data/keras_png_slices_seg_test")
# Pre process test labels data
test_Y = data.process_labels(test_Y)
 
# Check Images
pyplot.imshow(test_Y[2,:,:,3])
pyplot.show()