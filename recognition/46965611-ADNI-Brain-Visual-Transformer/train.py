"""
train.py

Code for training, validating, testing and saving model.

Author: Joshua Wang (Student No. 46965611)
Date Created: 11 Oct 2022
"""
from dataset import DataLoader
from modules import build_vision_transformer
import tensorflow as tf
import tensorflow_addons as tfa

# Hyperparameters
IMAGE_SIZE = 128
PATCH_SIZE = 16
BATCH_SIZE = 64
PROJECTION_DIM = 64
LEARNING_RATE = 0.001
ATTENTION_HEADS = 5
HIDDEN_UNITS = [PROJECTION_DIM * 2, PROJECTION_DIM]
DROPOUT_RATE = 0.1
TRANSFORMER_LAYERS = 5
INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

NUM_PATCHES = int((IMAGE_SIZE/PATCH_SIZE) ** 2)
CLASS_TYPES = ['NC', 'AD']
WEIGHT_DECAY = 0.0001
EPOCHS = 5

def train_model():
    # Load data
    loader = DataLoader("C:/AD_NC", IMAGE_SIZE, BATCH_SIZE)
    train, val, test = loader.load_data()

    # Build a compile model
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
        TRANSFORMER_LAYERS
    )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
        ]
    )

    # Train model
    history = model.fit(
        x=train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=val
    )
