"""
This is program for COMP3710 Report, resolving problem 6.
[Create a generative model of the OASIS brain or the OAI AKOA knee data set using a DCGAN that
has a “reasonably clear image” and a Structured Similarity (SSIM) of over 0.6.]

@auther: Ganze Zheng
@studentID: 44570776
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
# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(images[i])
# plt.show()

# Normalise data to [0, 1]
data = images.reshape((n_datasize, 256, 256, 1)).astype('float32')
data = data / 255.0
print(data.shape)

# Batch and shuffle the data
X_train = tf.data.Dataset.from_tensor_slices(
    data).shuffle(n_datasize).batch(batch_size)


# Build networks
# Generator network
def generator_network(input_shape, name='G'):
    """
    Generator network
    """
    input = Input(input_shape)
    dense = Dense(8*8*512)(input)
    norm = BatchNormalization()(dense)
    activation = LeakyReLU()(norm)
    dense = Reshape((8, 8, 512))(activation)
    # 8x8x512
    deepConvNet = Conv2DTranspose(
        256, (4, 4), strides=(2, 2), padding='same')(dense)
    norm = BatchNormalization()(deepConvNet)
    activation = LeakyReLU()(norm)
    # 16x16x256
    deepConvNet = Conv2DTranspose(
        128, (4, 4), strides=(2, 2), padding='same')(activation)
    norm = BatchNormalization()(deepConvNet)
    activation = LeakyReLU()(norm)
    # 32x32x128
    deepConvNet = Conv2DTranspose(
        64, (4, 4), strides=(2, 2), padding='same')(activation)
    norm = BatchNormalization()(deepConvNet)
    activation = LeakyReLU()(norm)
    # 64x64x64
    deepConvNet = Conv2DTranspose(
        32, (4, 4), strides=(2, 2), padding='same')(activation)
    norm = BatchNormalization()(deepConvNet)
    activation = LeakyReLU()(norm)
    # 128x128x32
    g_net = Conv2DTranspose(1, (4, 4), strides=(
        2, 2), padding='same', activation='sigmoid')(activation)
    # 256x256x1

    return Model(inputs=input, outputs=g_net, name=name)


# Discriminator network
def discriminator_network(input_shape, name='D'):
    """
    Discriminator network
    """
    input = Input(input_shape)
    # 256x256x1
    convNet = Conv2D(32, (4, 4), strides=(2, 2), padding='same')(input)
    activation = LeakyReLU()(convNet)
    norm = BatchNormalization()(activation)
    # 128x128x32
    convNet = Conv2D(64, (4, 4), strides=(2, 2), padding='same')(norm)
    activation = LeakyReLU()(convNet)
    norm = BatchNormalization()(activation)
    # 64x64x64
    convNet = Conv2D(128, (4, 4), strides=(2, 2), padding='same')(norm)
    activation = LeakyReLU()(convNet)
    norm = BatchNormalization()(activation)
    # 32x32x128
    convNet = Conv2D(256, (4, 4), strides=(2, 2), padding='same')(norm)
    activation = LeakyReLU()(convNet)
    norm = BatchNormalization()(activation)
    # 16x16x256

    flatten = Flatten()(norm)
    d_net = Dense(1, activation='sigmoid')(flatten)

    return Model(inputs=input, outputs=d_net, name=name)


# Build models
noise_shape = (100, )
image_shape = (256, 256, 1)

# Build generator
generator = generator_network(noise_shape)
generator.summary()
# Verify the model
noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)
plt.imshow(generated_image[0, :, :, 0])
plt.show()

# Build discriminator
discriminator = discriminator_network(image_shape)
discriminator.summary()
# Verify the model
decision = discriminator(generated_image)
print(decision)

print("End")
