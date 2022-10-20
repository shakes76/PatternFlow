import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import tensorflow as tf

from tensorflow.keras.optimizers import Adadelta, Nadam ,Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard


import dataset
import modules

modules.model.compile(optimizer='adam', loss='categorical_crossentropy' ,metrics=['accuracy'])

es = EarlyStopping(mode='max', monitor='val_acc', patience=10, verbose=0)
tb = TensorBoard(log_dir="logs/", histogram_freq=0, write_graph=True, write_images=False)
rl = ReduceLROnPlateau(monitor='val_acc',factor=0.1,patience=5,verbose=1,mode="max",min_lr=0.0001)

results = modules.model.fit(dataset.train_generator , steps_per_epoch=dataset.train_steps ,epochs=30,
                              validation_data=dataset.val_generator,validation_steps=dataset.val_steps,callbacks=[es,tb,rl])

'''
def show_subplot(original, reconstructed):
    plt.subplot(1, 2, 1)
    plt.imshow(original.squeeze() + 0.5,cmap='gray')
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed.squeeze() + 0.5,cmap='gray')
    plt.title("Reconstructed")
    plt.axis("off")

    plt.show()




import pickle

# same filename
filename = "resulta.pickle"
with open(filename, "rb") as file:
    x_train = pickle.load(file)

filename = "resultb.pickle"
with open(filename, "rb") as file:
    data_variance = pickle.load(file)

filename = "resultc.pickle"
with open(filename, "rb") as file:
    x_test_scaled = pickle.load(file)

'''