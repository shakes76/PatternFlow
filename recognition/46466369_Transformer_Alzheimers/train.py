import tensorflow as tf
from tensorflow import keras
from keras import layers
import dataset
import modules
import matplotlib.pyplot as plt
# images are of 256 x 240 size

IMAGE_SIZE = 64
PATCH_SIZE = 8
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
LEARNING_RATE = 0.001
NUM_EPOCH = 100
NUM_HEADS = 4
NUM_CLASSES = 2
PROJECTION_DIM = 64
TRANSFORMER_UNITS = [
    PROJECTION_DIM * 2,
    PROJECTION_DIM
]
MLP_HEAD_UNITS = [2048, 1024]

train, trainy, test, testy = dataset.load_dataset(IMAGE_SIZE)

model = modules.create_vit_classifier(PATCH_SIZE,
                                NUM_PATCHES,
                                PROJECTION_DIM,
                                MLP_HEAD_UNITS)


plt.imshow(train[1])
plt.show()

