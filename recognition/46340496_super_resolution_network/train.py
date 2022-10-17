import tensorflow as tf
import numpy as np
import math
from modules import *
from dataset import *

class ESPCNCallback(tf.keras.callbacks.Callback):
    test_ds = creating_test_dataset()

    def __init__(self):
        super(ESPCNCallback, self).__init__()
        for batch in test_ds.take(1):
            for img in batch:
                test_img = img
                break
        self.test_img = tf.image.resize(test_img, (64, 60))

    # Store PSNR value in each epoch
    def on_epoch_begin(self, epoch, logs=None):
        self.psnr = []

    def on_epoch_end(self, epoch, logs=None):
        print("Mean PSNR for epoch: %.2f" %  (np.mean(self.psnr)))
        # model, callbacks, loss_fn, optimizer = model_checkpoint()
        if True:
            result = model(self.test_img[tf.newaxis, ...])[0]
            plt.imshow(result)
            plt.show()

    def on_test_batch_end(self, batch, logs=None):
        self.psnr.append(10 * math.log10(1 / logs["loss"]))

# def model_checkpoint():

model = get_model(upscale_factor=UPSCALE_FACTOR, channels=1)
model.summary()

callbacks = [ESPCNCallback()]
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # return model, callbacks, loss_fn, optimizer

# def training():

    # model, callbacks, loss_fn, optimizer = model_checkpoint()

epochs = 5
train_ds, valid_ds = mapping_target()
        
model.compile(
    optimizer=optimizer, loss=loss_fn,
)

model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=valid_ds, verbose=1
)

# sup = training()