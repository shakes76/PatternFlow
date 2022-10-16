from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import layers


def get_training_data(classes, model):
    x_train, x_test, y_train, y_test = train_test_split(
        model,
        classes,
        test_size=0.9,
    )
    return x_train, x_test, y_train, y_test


def setup_training(training_data):
    x_train, x_test, y_train, y_test = training_data


def handle_training(sorted_data):
    classes, model = sorted_data
    training_data = get_training_data(classes, model)
    setup_training(training_data)
