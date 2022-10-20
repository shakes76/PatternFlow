from dataset import get_datasets
from modules import get_model

from pkgutil import get_data
import numpy as np
import tensorflow as tf
import tensorflow.keras.utils
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import PIL


def train():
    train_path = "dataset\\train"
    test_path = "dataset\\test"
    
    batch_size = 8
    upscale_factor = 4
    crop_size = 200
    train_ds, val_ds, test_ds = get_datasets(train_path, test_path, batch_size, upscale_factor, crop_size)
    checkpoint_filepath = "tmp\\checkpoint\\"

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor="loss",
    mode="min",
    save_best_only=True,
    )
    
    
    
    model = get_model()

    loss_fn = keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    epochs = 100
    model.compile(
        optimizer=optimizer, loss=loss_fn,
    )

    model.fit(
        train_ds, epochs=epochs, callbacks = [model_checkpoint_callback],  validation_data=val_ds, verbose=2
    )

    model.load_weights(checkpoint_filepath)

def plot_results(img, prefix, title):
    img_array = img_to_array(img)
    img_array = img_array.astype('float32') / 255.0

    fig, ax = plt.subplots()
    im = ax.imshow(img_array[::-1], origin="lower")
    
    plt.title(title)

    plt.yticks(visible=False)
    plt.xticks(visible=False)
    plt.savefig(str(prefix) + "-" + title + ".png")
    plt.show()

def get_lowres_image(img, upscale_factor):
    return tf.imgage.resize(
        img,
        (img.size[0] // upscale_factor, img.size[1] // upscale_factor),
        PIL.Image.BICUBIC
    )

def main():
    train()

if __name__ == "__main__":
    main()