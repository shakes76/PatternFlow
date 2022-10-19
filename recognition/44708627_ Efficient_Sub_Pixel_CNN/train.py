#!/usr/bin/env python
# coding: utf-8

# In[4]:


import tensorflow as tf

# import os
import math
import numpy as np

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import img_to_array

from IPython.display import display

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import PIL


# In[5]:


from ipynb.fs.full.dataset import *
from ipynb.fs.full.moduel import *


# In[6]:


train_dir = "./dataset/train"
test_dir = "./dataset/test"
test_img_paths = test_imgs(test_dir)
train_ds, valid_ds = setup_dataset(train_dir)
train_ds, valid_ds = dataset_preprocessing(train_ds,valid_ds)


# In[7]:


# def plot_results(img, prefix, title):
#     """Plot the result with zoom-in area."""
#     img_array = img_to_array(img)
#     img_array = img_array.astype("float32") / 255.0

#     # Create a new figure with a default 111 subplot.
#     fig, ax = plt.subplots()
#     im = ax.imshow(img_array[::-1], origin="lower")

#     plt.title(title)
#     # zoom-factor: 3, location: upper-left
#     axins = zoomed_inset_axes(ax, 4, loc=3)
#     axins.imshow(img_array[::-1], origin="lower")

#     # Specify the limits.
#     x1, x2, y1, y2 = 75, 160, 165, 180
#     # Apply the x-limits.
#     axins.set_xlim(x1, x2)
#     # Apply the y-limits.
#     axins.set_ylim(y1, y2)

#     plt.yticks(visible=False)
#     plt.xticks(visible=False)

#     # Make the line.
#     mark_inset(ax, axins, loc1=2, loc2=1, fc="none", ec="blue")
# #     plt.savefig(str(prefix) + "-" + title + ".png")
#     plt.show()


def get_lowres_image(img, upscale_factor):
    """Return low-resolution image to use as model input."""
    return img.resize(
        (img.size[0] // upscale_factor, img.size[1] // upscale_factor),
        PIL.Image.BICUBIC,
    )


def upscale_image(model, img):
    """Predict the result based on input image and restore the image as RGB."""
    ycbcr = img.convert("YCbCr")
    y, cb, cr = ycbcr.split()
    y = img_to_array(y)
    y = y.astype("float32") / 255.0

    input = np.expand_dims(y, axis=0)
    out = model.predict(input)

    out_img_y = out[0]
    out_img_y *= 255.0

    # Restore the image in RGB color space.
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = out_img_y.reshape((np.shape(out_img_y)[0], np.shape(out_img_y)[1]))
    out_img_y = PIL.Image.fromarray(np.uint8(out_img_y), mode="L")
    out_img_cb = cb.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img = PIL.Image.merge("YCbCr", (out_img_y, out_img_cb, out_img_cr)).convert(
        "RGB"
    )
    return out_img


# In[8]:


class ESPCNCallback(keras.callbacks.Callback):
    def __init__(self):
        super(ESPCNCallback, self).__init__()
        self.test_img = get_lowres_image(load_img(test_img_paths[0]), upscale_factor)

    # Store PSNR value in each epoch.
    def on_epoch_begin(self, epoch, logs=None):
        self.psnr = []

    def on_epoch_end(self, epoch, logs=None):
        print("Mean PSNR for epoch: %.2f" % (np.mean(self.psnr)))
#         if epoch % 20 == 0:
#             prediction = upscale_image(self.model, self.test_img)
#             plot_results(prediction, "epoch-" + str(epoch), "prediction")

    def on_test_batch_end(self, batch, logs=None):
        self.psnr.append(10 * math.log10(1 / logs["loss"]))


# In[9]:


# Stop training when a monitored metric has stopped improving.
early_stopping_callback = keras.callbacks.EarlyStopping(monitor="loss", patience=10)

checkpoint_filepath = "./model_weights/ckpt"

# save a model or weights (in a checkpoint file) at some interval, 
# so the model or weights can be loaded later to continue the 
# training from the state saved.

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor="loss",
    mode="min",
    save_best_only=True,
)

model = get_model(upscale_factor=upscale_factor, channels=1)
model.summary()

callbacks = [ESPCNCallback(), early_stopping_callback, model_checkpoint_callback]
loss_fn = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam(learning_rate=0.001)


# In[7]:


epochs = 100

model.compile(
    optimizer=optimizer, loss=loss_fn,
)

model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=valid_ds, verbose=2
)

# The model weights (that are considered the best) are loaded into the model.
# model.load_weights(checkpoint_filepath)

