from tensorflow import keras
import tensorflow_addons as tfa
from modules import vit_classifier
from dataset import import_data
import matplotlib.pyplot as plt
from config import *


# import test data
paths = {"training": path_training, "validation": path_validation, "test": path_test} 

_, _, data_test = import_data(IMAGE_SIZE, BATCH_SIZE, paths)

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
checkpoint_filepath = "C:\\Users\\lovet\\Documents\\COMP3710\\Report\\adni\\checkpoint2\\"


# evaluate the model 
vit_classifier.load_weights(checkpoint_filepath)
_, accuracy, = vit_classifier.evaluate(x=data_test)
print(f"Test accuracy: {round(accuracy * 100, 2)}%")