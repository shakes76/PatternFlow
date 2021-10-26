import tensorflow as tf
from functools import partial

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 96
IMAGE_CHANNELS = 3

class Isic_Unet:
    def __init__(self):
        self.model = None
    
    def improved_unet(self):
        dropout = 0.1
        # encoder/downsampling
        input_size = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)
        inputs = tf.keras.Input(input_size)
        conv1 = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation=partial(tf.nn.leaky_relu, alpha=0.01))(inputs)
        conv1 = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation=partial(tf.nn.leaky_relu, alpha=0.01))(conv1)
        pool1 = tf.keras.layers.MaxPool2D((2, 2))(conv1)
        pool1 = tf.keras.layers.Dropout(dropout)(pool1)

        conv2 = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation=partial(tf.nn.leaky_relu, alpha=0.01))(pool1)
        conv2 = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation=partial(tf.nn.leaky_relu, alpha=0.01))(conv2)
        pool2 = tf.keras.layers.MaxPool2D((2, 2))(conv2)
        pool2 = tf.keras.layers.Dropout(dropout)(pool2)
        
        conv3 = tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation=partial(tf.nn.leaky_relu, alpha=0.01))(pool2)
        conv3 = tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation=partial(tf.nn.leaky_relu, alpha=0.01))(conv3)
        pool3 = tf.keras.layers.MaxPool2D((2, 2))(conv3)
        pool3 = tf.keras.layers.Dropout(dropout)(pool3)
        
        conv4 = tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation=partial(tf.nn.leaky_relu, alpha=0.01))(pool3)
        conv4 = tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation=partial(tf.nn.leaky_relu, alpha=0.01))(conv4)
        pool4 = tf.keras.layers.MaxPool2D((2, 2))(conv4)
        pool4 = tf.keras.layers.Dropout(dropout)(pool4)
        
        # bridge/bottleneck/shared layer
        conv5 = tf.keras.layers.Conv2D(1024, (3, 3), padding="same", activation=partial(tf.nn.leaky_relu, alpha=0.01))(pool4)
        conv5 = tf.keras.layers.Conv2D(1024, (3, 3), padding="same", activation=partial(tf.nn.leaky_relu, alpha=0.01))(conv5)

        # decoder/upsampling
        up6 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv5)
        up6 = tf.keras.layers.Conv2D(512, (2, 2), padding="same")(up6)
        up6 = tf.keras.layers.concatenate([conv4, up6])
        conv6 = tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation=partial(tf.nn.leaky_relu, alpha=0.01))(up6)
        conv6 = tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation=partial(tf.nn.leaky_relu, alpha=0.01))(conv6)
        conv6 = tf.keras.layers.Dropout(dropout)(conv6)

        up7 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv6)
        up7 = tf.keras.layers.Conv2D(256, (2, 2),  padding="same")(up7)
        up7 = tf.keras.layers.concatenate([conv3, up7])
        conv7 = tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation=partial(tf.nn.leaky_relu, alpha=0.01))(up7)
        conv7 = tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation=partial(tf.nn.leaky_relu, alpha=0.01))(conv7)
        conv7 = tf.keras.layers.Dropout(dropout)(conv7)

        up8 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv7)
        up8 = tf.keras.layers.Conv2D(128, (2, 2), padding="same")(up8)
        up8 = tf.keras.layers.concatenate([conv2, up8])
        conv8 = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation=partial(tf.nn.leaky_relu, alpha=0.01))(up8)
        conv8 = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation=partial(tf.nn.leaky_relu, alpha=0.01))(conv8)
        conv8 = tf.keras.layers.Dropout(dropout)(conv8)

        up9 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv8)
        up9 = tf.keras.layers.Conv2D(64, (2, 2), padding="same")(up9)
        up9 = tf.keras.layers.concatenate([conv1, up9])
        conv9 = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation=partial(tf.nn.leaky_relu, alpha=0.01))(up9)
        conv9 = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation=partial(tf.nn.leaky_relu, alpha=0.01))(conv9)
        conv9 = tf.keras.layers.Dropout(dropout)(conv9)

        # segmentation (output) layer
        outputs = tf.keras.layers.Conv2D(1, (1, 1), padding="same", activation='softmax')(conv9)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)


    def show_predictions(self):
        """
        Perform prediction on validation set and report performance
        Based on code from COMP3710-demo-code.ipynb from Guest Lecture.
        """

        # generate predicted masks
        image_batch, mask_batch = next(iter(self.val_ds.batch(3)))
        predictions = self.model.predict(image_batch)

        # visualise images and masks
        import matplotlib.pyplot as plt
        plt.figure(figsize=(20, 10))
        for i in range(3):
            # show base image
            plt.subplot(3, 3, 3*i+1)
            plt.imshow(image_batch[i])
            plt.axis('off')

            # show true mask
            plt.subplot(3, 3, 3*i+2)
            plt.imshow(mask_batch[i])
            plt.axis('off')

            # show predicted mask
            plt.subplot(3, 3, 3*i+3)
            plt.imshow(predictions[i])
            plt.axis('off')
        plt.show()    