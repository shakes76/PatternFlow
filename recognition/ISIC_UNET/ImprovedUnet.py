# https://echo360.org.au/lesson/G_c55b0d11-15c0-4b87-b69d-7410edfb5a76_38ecd661-9fca-4df3-9d60-51f440fd1f5c_2021-09-20T12:00:00.000_2021-09-20T14:00:00.000/classroom#sortDirection=desc
# pseudo code
# conv2d relu twice, max pool, do this 4 times; then conv1x1 twice ad then up-conv2x2.
# then conv2d relu and up conv4 times and on the last time just conv2d twice and then conv1x1.
# this is the typical UNet

# data: fetaures (images), labels ( segmentatuon masks)
# preprocessing: normalise the images, one hot encode the labels - split mask image into multiple ones with 0 and 1's
# use softmax (not sigmoid) for unet
# need to define loss model (dice loss in this case) - ranges from 0 to 1 where 1 is best, DSC = (2|X intersect Y|)/(|X|+|Y|)
# have to report the dice for each class

# class notes
# unet, isic you have 2 types of images, demoscopy and the mask
# demoscopy is the input, mask is the output (labels)
# mask is binary image (black background, white is legion etc.)
# binary classification at the pixel level, classify each pixel as background or legion
#
# ground truth is the mask
# demoscopy is the input, train it with the mask as the output
#
# binary use the sigmoid
#
# look at the improved unet paper linked
#
# they just have additional residual connections
# paper is 3d, just make 2d, just need to implement the residual ones
#
# different size of images, just resize
#
# look at github to see sample solutions
#
# dice is the loss in the paper too, it has formula too
#
# watch lecture


import tensorflow as tf


def dsc_loss(ytrue, ypred):
    true_flat = tf.keras.backend.flatten(ytrue)
    pred_flat = tf.keras.backend.flatten(ypred)
    inter = tf.keras.backend.sum(true_flat * pred_flat)
    return 1 - (2*(inter + 1) / (tf.reduce_sum(true_flat) + tf.reduce_sum(pred_flat) + 1))

def model():

    inputs = tf.keras.Input((256, 256, 1))
    # step 1
    conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu")(inputs)
    conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu")(conv1)
    pool1 = tf.keras.layers.MaxPool2D((2, 2))(conv1)

    conv2 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu")(pool1)
    conv2 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu")(conv2)
    pool2 = tf.keras.layers.MaxPool2D((2, 2))(conv2)

    conv3 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", activation="relu")(pool2)
    conv3 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", activation="relu")(conv3)
    pool3 = tf.keras.layers.MaxPool2D((2, 2))(conv3)

    conv4 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", activation="relu")(pool3)
    conv4 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", activation="relu")(conv4)
    pool4 = tf.keras.layers.MaxPool2D((2, 2))(conv4)

    conv5 = tf.keras.layers.Conv2D(filters=1024, kernel_size=3, padding="same", activation="relu")(pool4)
    conv5 = tf.keras.layers.Conv2D(filters=1024, kernel_size=3, padding="same", activation="relu")(conv5)

    # up sampling
    up6 = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=3, padding="same", activation="relu", strides=(2,2))(conv5)
    m6 = tf.keras.layers.concatenate([conv4, up6], axis=3)
    conv6 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", activation="relu")(m6)
    conv6 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", activation="relu")(conv6)

    up7 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=3, padding="same", activation="relu", strides=(2,2))(conv6)
    m7 = tf.keras.layers.concatenate([conv3, up7], axis=3)
    conv7 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", activation="relu")(m7)
    conv7 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", activation="relu")(conv7)

    up8 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, padding="same", activation="relu", strides=(2,2))(conv7)
    m8 = tf.keras.layers.concatenate([conv2, up8], axis=3)
    conv8 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu")(m8)
    conv8 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu")(conv8)

    up9 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, padding="same", activation="relu", strides=(2,2))(conv8)
    m9 = tf.keras.layers.concatenate([conv1, up9], axis=3)
    conv9 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu")(m9)
    conv9 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu")(conv9)

    conv10 = tf.keras.layers.Conv2D(filters=1, kernel_size=1, activation='sigmoid')(conv9)

    model = tf.keras.Model(inputs, conv10)


    model.compile(tf.keras.optimizers.Adam(), loss=dsc_loss, metrics=['accuracy'])
    model.summary()

model()
