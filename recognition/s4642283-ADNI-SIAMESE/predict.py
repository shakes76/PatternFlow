import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import sys

# Takes a path to data as a command line argument and makes predictions.
data_path = sys.argv[1]

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

# Load Model
classifier_model = keras.models.load_model(os.path.join(__location__, "Classifier_Model"))

X_data = []
X_data_labels = []
# Load Data (Expects two directory at location, an AD (Alzheimer's) and a NC directory)
print(os.path.join(__location__, data_path, "AD"))
for fname in os.listdir(os.path.join(__location__, data_path, "AD")):
    fpath = os.path.join(__location__, data_path, "AD", fname)
    im = Image.open(fpath)
    X_data.append(np.array(im))
    X_data_labels.append(1)
    im.close()

for fname in os.listdir(os.path.join(__location__, data_path,"NC")):
    fpath = os.path.join(__location__, data_path, "NC", fname)
    im = Image.open(fpath)
    X_data.append(np.array(im))
    X_data_labels.append(0)
    im.close()

# Convert to numpy array
X_data = np.array(X_data)
X_data_labels = np.array(X_data_labels)

# Make predictions on given data
classifier_model.evaluate(X_data, X_data_labels)