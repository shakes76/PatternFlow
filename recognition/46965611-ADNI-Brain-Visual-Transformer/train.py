"""
train.py

Code for training, validating, testing and saving model.

Author: Joshua Wang (Student No. 46965611)
Date Created: 11 Oct 2022
"""
from dataset import load_data
from modules import build_vision_transformer
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from parameters import *


def compile_model():
    """
    Builds and compiles the model.
    """
    # Build and compile model
    optimizer = tfa.optimizers.AdamW(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    model = build_vision_transformer(
        INPUT_SHAPE,
        IMAGE_SIZE,
        PATCH_SIZE,
        NUM_PATCHES,
        ATTENTION_HEADS,
        PROJECTION_DIM,
        HIDDEN_UNITS,
        DROPOUT_RATE,
        TRANSFORMER_LAYERS,
        MLP_HEAD_UNITS
    )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy')
        ]
    )

    return model
    

def train_model(model, train_data, val_data):
    """
    Trains and saves the model.
    """

    # Train model
    history = model.fit(
        x=train_data,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=val_data
    )

    # Save model
    model.save(
        MODEL_SAVE_PATH,
        overwrite=True,
        include_optimizer=True,
        save_format='tf'
    )

    # Plot and save accuracy curves
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.suptitle('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'])
    plt.savefig('accuracy.png')
    plt.clf()

    # Plot and save loss curves
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.suptitle('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'])
    plt.savefig('losses.png')
    plt.clf()


if __name__ == '__main__':
    # Load data
    train, val, test = load_data()

    # Compile and train model
    model = compile_model()
    print(model.summary())
    train_model(model, train, val)