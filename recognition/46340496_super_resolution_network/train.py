from pyexpat import model
import tensorflow as tf
import numpy as np
import math
from modules import *
from dataset import *
class ESPCNCallback(tf.keras.callbacks.Callback):

    test_ds = creating_test_dataset()

    def __init__(self):
        super(ESPCNCallback, self).__init__()
        self.test_img = get_lowres_image(tf.keras.preprocessing.image.load_img(test_ds[0]), UPSCALE_FACTOR)

    # Store PSNR value in each epoch
    def on_epoch_begin(self, epoch, logs=None):
        self.psnr = []

    def on_epoch_end(self, epoch, logs=None):
        print("Mean PSNR for epoch: %.2f" %  (np.mean(self.psnr)))
        if epoch % 20 == 0:
            prediction = upscale_image(self.model, self.test_img)
            plot_results(prediction, "epoch-" + str(epoch), "prediction")

    def on_test_batch_end(self, batch, logs=None):
        self.psnr.append(10 * math.log10(1 / logs["loss"]))

def model_checkpoint():
    # early stopping
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=10)

    checkpoint_filepath = r"C:\Users\galla\OneDrive\University\Year 3\Semester 2\COMP3710\Repor\tmp\checkpoint"

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath = checkpoint_filepath,
        save_weights_only=True,
        monitor="loss",
        mode="min",
        save_best_only=True,
    )

    model = get_model(upscale_factor=UPSCALE_FACTOR, channels=1)
    model.summary()

    callbacks = [ESPCNCallback(), early_stopping_callback, model_checkpoint_callback]
    loss_fn = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    return model, callbacks, loss_fn, optimizer, checkpoint_filepath

def training():

    model, callbacks, loss_fn, optimizer, checkpoint_filepath = model_checkpoint()

    epochs = 5
    train_ds, valid_ds = mapping_target()

    # for batch in valid_ds.take(1):
    #     for img in batch[0]:
    #         print("valid", img)
    #         img_plot = plt.imshow(img)
    #         plt.show()
    #     for img in batch[1]:
    #         print("valid", img)
    #         img_plot = plt.imshow(img)
    #         plt.show()

    # # Visualise input and target
    # for batch in train_ds.take(1):
    #     for img in batch[0]:
    #         print(img)
    #         img_plot = plt.imshow(img)
    #         plt.show()
    #     for img in batch[1]:
    #         print(img)
    #         img_plot = plt.imshow(img)
    #         plt.show()
        
    model.compile(
        optimizer=optimizer, loss=loss_fn,
    )

    model.fit(
        train_ds, epochs=epochs, callbacks=callbacks, validation_data=valid_ds, verbose=1
    )

    model.load_weights(checkpoint_filepath)

sup = training()