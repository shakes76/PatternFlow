"""
This is program for COMP3710 Report, resolving problem 6.
[Create a generative model of the OASIS brain or the OAI AKOA knee data set using a DCGAN that
has a “reasonably clear image” and a Structured Similarity (SSIM) of over 0.6.]
"""

import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Activation, LeakyReLU, Conv2D, Conv2DTranspose, BatchNormalization, Flatten, Dense, Reshape
from tensorflow.keras.models import Model
from PIL import Image

# parameters
epochs = 50
batch_size = 128
path = "../keras_png_slices_data/keras_png_slices_train/*"

# Load OASIS Dataset
filenames = glob.glob(path)

if len(filenames) == 0:
    print("Error! Images not loaded!")
    exit()
else:
    n_datasize = len(filenames)

images = list()

for i in range(n_datasize):
    images.append(np.asarray(Image.open(filenames[i])))

images = np.array(images)

# Check dataset shape and verify the images
print(images.shape)
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(images[i])
plt.show()

# Normalise data to [-1, 1]
data = images.reshape((n_datasize, 256, 256, 1)).astype('float32')
data = (data - 127.5) / 127.5
print(data.shape)

# Batch and shuffle the data
X_train = tf.data.Dataset.from_tensor_slices(data).shuffle(n_datasize).batch(batch_size)


# Build networks
# Generator network
def generator_network(input_shape):
    """
    Generator network
    """
    input = Input(input_shape)
    dense = Dense(8*8*512)(input)
    dense = Reshape((8, 8, 512))(dense)
    # 8x8x512
    net = Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same')(dense)
    norm = BatchNormalization()(net)
    activation = LeakyReLU()(norm)
    # 16x16x256
    net = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(activation)
    norm = BatchNormalization()(net)
    activation = LeakyReLU()(norm)
    # 32x32x128
    net = Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same')(activation)
    norm = BatchNormalization()(net)
    activation = LeakyReLU()(norm)
    # 64x64x64
    net = Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same')(activation)
    norm = BatchNormalization()(net)
    activation = LeakyReLU()(norm)
    # 256x256x3
    g_net = Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', activation='tanh')(activation)

    return Model(inputs=input, outputs=g_net)


# Build models
input_shape = (1, 1, 100)

# Build generator
generator = generator_network(input_shape)
generator.summary()
# Verify the model
noise = tf.random.normal([1, 1, 1, 100])
generated_images = generator(noise, training=False)
plt.imshow(generated_images[0, :, :, 0])
plt.show()

print("End")
