import math
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

def upscale_image(model, img):
    # The model takes a 4 dimensional array with shape (Batch, x, y, channels)
    # the img has shape (x, y, channels) so needs to be expanded
    input = np.expand_dims(img, axis=0)
    out = model.predict(input)
    # Now the output needs to go back to having the shape it had before
    prediction = out[0]
    prediction = prediction.clip(0, 255)
    return prediction

def train(epochs):
    train_path = "dataset\\train"
    test_path = "dataset\\test"
    
    batch_size = 8
    upscale_factor = 4
    crop_size = 200
    train_ds, val_ds, test_ds = get_datasets(train_path, test_path, batch_size, upscale_factor, crop_size)
    checkpoint_filepath = "tmp\\checkpoint"

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor="loss",
        mode="min",
        save_best_only=True,
    )
    
    class ESPCNCallback(keras.callbacks.Callback):
        def __init__(self):
            super(ESPCNCallback, self).__init__()
            self.test_img = get_lowres_image(list(test_ds.take(1))[0][0], upscale_factor)

        # Store PSNR value in each epoch.
        def on_epoch_begin(self, epoch, logs=None):
            self.psnr = []

        def on_epoch_end(self, epoch, logs=None):
            print("Mean PSNR for epoch: %.2f" % (np.mean(self.psnr)))
            if epoch % 20 == 0:
                prediction = upscale_image(self.model, self.test_img)
                plot_results(prediction, "epoch-" + str(epoch), "prediction")

        def on_test_batch_end(self, batch, logs=None):
            self.psnr.append(10 * math.log10(1 / logs["loss"]))
    
    model = get_model()

    loss_fn = keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.load_weights(checkpoint_filepath)
    model.compile(
        optimizer=optimizer, loss=loss_fn,
    )

    history = model.fit(
        train_ds, epochs=epochs, callbacks = [model_checkpoint_callback, ESPCNCallback()],  validation_data=val_ds, verbose=2
    )

    

    return model, test_ds, history

def plot_results(img, prefix, title):
    img_array = img_to_array(img)
    img_array = img_array.astype('float32') / 255.0

    fig, ax = plt.subplots()
    im = ax.imshow(img_array, origin="lower", cmap="gray")
    
    plt.title(title)

    plt.yticks(visible=False)
    plt.xticks(visible=False)
    plt.savefig("images/" + str(prefix) + "-" + title + ".png")
    # plt.show()

def get_lowres_image(img, upscale_factor):
    return tf.image.resize(
        img,
        (img.shape[0] // upscale_factor, img.shape[1] // upscale_factor)
    )

def main():
    epochs = 100
    _, _, history = train(epochs)
    print(history.history.keys())
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    xs = range(epochs)
    plt.figure()
    plt.plot(xs, loss, label = "loss")
    plt.plot(xs, val_loss, label = "val_loss")
    plt.title("Loss and Val_loss Over Time")
    plt.legend()
    plt.show()
    plt.savefig("images/losses.png")


if __name__ == "__main__":
    main()