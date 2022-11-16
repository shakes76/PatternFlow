"""train.py

The source code for traiing, validating, testing, and saving the model.
"""

from constants import BATCH_SIZE
from modules import get_model, ESPCNCallback
from dataset import download_data, get_datasets, get_tuple_from_dataset, \
    preview_data
from predict import display_predictions

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


def run_model(epochs: int = 30) -> keras.Model:
    """Download data and return a trained model

    Args:
        epochs (int, optional): Number of epochs to train for. Defaults to 30.

    Returns:
        keras.Model: The trained model
    """
    data_dir = download_data()
    print(f"Data downloaded to {data_dir}")

    train_ds, test_ds = get_datasets(data_dir)
    model = train_model(train_ds, test_ds, epochs)

    print("Displaying prediction images using the trained model...")
    display_predictions(test_ds, model)

    return model


def train_model(
    train_ds: tf.data.Dataset,
    test_ds: tf.data.Dataset,
    epochs: int,
) -> keras.Model:
    """Train the super-resolution model, saving the best checkpoint

    Args:
        train_ds (tf.data.Dataset): high-res training dataset
        test_ds (tf.data.Dataset): high-res testing dataset
        epochs (int): number of epochs to train for

    Returns:
        keras.Model: The trained model
    """

    preview_data(train_ds, "Training dataset: downsampled image vs target")
    preview_data(test_ds, "Testing dataset: downsampled image vs target")

    model = get_model()
    model.summary()

    _, test_image = get_tuple_from_dataset(test_ds)

    callbacks = [ESPCNCallback(test_image)]
    loss_fn = keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
    )

    history = model.fit(
        train_ds,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        validation_data=test_ds,
        verbose=1,
    )

    # Plot loss
    plt.figure(figsize=(15, 10))
    plt.plot(history.history["loss"], label="Loss")
    plt.plot(history.history["val_loss"], "--", label="Validation Loss")
    plt.title("Training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    return model
