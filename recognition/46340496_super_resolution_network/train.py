import tensorflow as tf
import numpy as np
import math
from modules import *
from dataset import *

# Class for callback and printing out PSNR values
class ESPCNCallback(tf.keras.callbacks.Callback):

    def __init__(self):
        test_ds = creating_test_dataset()
        super(ESPCNCallback, self).__init__()
        for batch in test_ds.take(1):
            for img in batch:
                test_img = img
                break
        self.test_img = tf.image.resize(test_img, (64, 60))
        self.model = get_model(upscale_factor=UPSCALE_FACTOR, channels=1)
        self.train_psnr = []

    # Store PSNR value in each epoch
    def on_epoch_begin(self, epoch, logs=None):
        self.psnr = []

    def on_epoch_end(self, epoch, logs=None):
        print("Mean PSNR for epoch: %.2f" %  (np.mean(self.psnr)))
        self.train_psnr.append(np.mean(self.psnr))
        # if True:
        #     result = self.model(self.test_img[tf.newaxis, ...])[0]
        #     plt.imshow(result)
        #     plt.show()

    def on_test_batch_end(self, batch, logs=None):
        self.psnr.append(10 * math.log10(1 / logs["loss"]))

    def on_train_end(self, logs=None):
        plt.plot(self.train_psnr)


# Creates the varaibles that are required for training the model
def model_checkpoint():

    model = get_model(upscale_factor=UPSCALE_FACTOR, channels=1)
    model.summary()

    callbacks = [ESPCNCallback()]
    loss_fn = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    return model, callbacks, loss_fn, optimizer

# The training of the model
def training():

    model, callbacks, loss_fn, optimizer = model_checkpoint()

    epochs = 5
    train_ds, valid_ds = mapping_target()
            
    model.compile(
        optimizer=optimizer, loss=loss_fn,
    )

    history = model.fit(
        train_ds, epochs=epochs, callbacks=callbacks, validation_data=valid_ds, verbose=1
    )

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    return model, loss, val_loss

