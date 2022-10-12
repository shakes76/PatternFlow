"""train.py

The source code for traiing, validating, testing, and saving the model.
"""

from constants import CHECKPOINT_FILEPATH, BATCH_SIZE
from modules import get_model, ESPCNCallback
from dataset import download_data, get_datasets, downsample_data, \
    get_image_from_dataset, preview_data
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

    print("Displaying prediction images using the model...")
    display_predictions(test_ds, model)

    return model


def train_model(
    train_ds: tf.data.Dataset,
    test_ds: tf.data.Dataset,
    epochs: int,
    checkpoint: str | None = None,
) -> keras.Model:
    """Train the super-resolution model, saving the best checkpoint

    Args:
        train_ds (tf.data.Dataset): high-res training dataset
        test_ds (tf.data.Dataset): high-res testing dataset
        epochs (int): number of epochs to train for
        checkpoint (str | None): Optional path to a checkpoint file. If given,
            the model will use the checkpoint specified instead of training from
            the beginning. Defaults to None.

    Returns:
        keras.Model: The trained model
    """
    down_train_ds = downsample_data(train_ds)
    down_test_ds = downsample_data(test_ds)

    preview_data(down_train_ds, "Training dataset: downsampled image vs target")
    preview_data(down_test_ds, "Testing dataset: downsampled image vs target")

    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor="loss",
        patience=10
    )

    checkpoint_filename = CHECKPOINT_FILEPATH + "-{epoch:04d}.ckpt"

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filename,
        save_weights_only=True,
        verbose=1,
    )

    model = get_model()
    model.summary()

    if checkpoint:
        model = model.load_weights(checkpoint)
        print(f"Loaded checkpoint weights from {checkpoint} into model")

    test_image = get_image_from_dataset(test_ds)

    callbacks = [
        ESPCNCallback(test_image),
        early_stopping_callback,
        model_checkpoint_callback
    ]
    loss_fn = keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
    )

    history = model.fit(
        down_train_ds,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        validation_data=down_test_ds,
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
