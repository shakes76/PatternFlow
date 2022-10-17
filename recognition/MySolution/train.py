import stellargraph as sg
from tensorflow import keras
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pandas as pd
import keras.losses
import tensorflow as tf


def plot_loss_epoch(history):
    plt.plot(history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper right')
    plt.show()


def handle_training(module, new_model=False):
    train_gen = module.get_train_gen()
    train_data, val_data, test_data, train_targets, val_targets, test_targets =\
        module.get_data_group()
    model = module.get_model()
    history_log = keras.callbacks.CSVLogger(
        "history_log.csv",
        separator=",",
        append=True
    )
    history = model.fit(
        train_gen,
        epochs=5,
        verbose=2,
        callbacks=[history_log]
    )
    model.save("pre-trained_model")
    plot_loss_epoch(history.history)



