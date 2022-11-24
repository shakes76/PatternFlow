import tensorflow as tf
from tensorflow import keras
from keras import layers
import dataset
import modules
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
# images are of 256 x 240 size

IMAGE_SIZE = 128
PATCH_SIZE = 8
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
BATCH_SIZE = 256
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0015
NUM_EPOCH = 100
NUM_HEADS = 4
NUM_CLASSES = 2
PROJECTION_DIM = 64
TRANSFORMER_UNITS = [
    PROJECTION_DIM * 2,
    PROJECTION_DIM
]
TRANSFORMER_LAYERS = 8
MLP_LAYER_COUNTS = [2048, 1024]

file_path = './checkpoint'

train, trainy, test, testy = dataset.load_dataset(IMAGE_SIZE)

vit_model = modules.create_vit_classifier(PATCH_SIZE,
                                train,
                                NUM_PATCHES,
                                PROJECTION_DIM,
                                NUM_CLASSES,
                                NUM_HEADS,
                                TRANSFORMER_LAYERS,
                                TRANSFORMER_UNITS,
                                MLP_LAYER_COUNTS)

def train_model(model):
    
    optimizer = tfa.AdamW(learning_rate=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)
    model.compile(optimizer=optimizer, 
                    loss=tf.losses.sparse_categorical_crossentropy(), 
                    metrics=[tf.metrics.sparse_categorical_accuracy()])

    callback = tf.keras.callbacks.ModelCheckpoint(file_path,
                                                    monitor="val_accuracy",
                                                    save_best_only=True,
                                                    save_weights_only=True)
    
    train_history = model.fit(x=train,
                                y=trainy,
                                batch_size=BATCH_SIZE,
                                epochs=NUM_EPOCH,
                                validation_split=0.1,
                                callbacks=[callback])

    return (train_history)


def test_model(model):
    model.load_weights(file_path)
    _, accuracy = model.evaluate(test, testy)
    print(f'Test accuracy: {round(accuracy * 100, 2)}%')

train_model(vit_model)
test_model(vit_model)
