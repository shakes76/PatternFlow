import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers

"""
    dice coef function to compute the dice score during training.by using argmax such that it is not continuous and 
    will not give derivatives instead it computes the true dice score.
"""


def dice_coef(y_true, y_pred):
    y_true_am = K.argmax(y_true, axis=3)
    y_pred_am = K.argmax(y_pred, axis=3)
    intersection = 2 * K.cast(K.sum(y_true_am * y_pred_am, tuple(range(1, len(y_pred_am.shape)))), 'float32')
    total = K.cast(K.sum(y_true_am + y_pred_am, tuple(range(1, len(y_pred_am.shape)))), 'float32')

    return K.mean(intersection / total)


"""
    dice_coef_loss which provides a loss form of the dice coefficient computed from the probabilistic
    values in the network. it allow the network to do the Gradient descent
"""


def dice_coef_loss(y_true, y_pred, epsilon=1e-5):
    intersection = 2 * K.cast(K.sum(y_true * y_pred, tuple(range(1, len(y_pred.shape) - 1))), 'float32')
    total = K.cast(K.sum(K.square(y_true) + K.square(y_pred), tuple(range(1, len(y_pred.shape) - 1))), 'float32')

    return 1 - K.mean((intersection + epsilon) / (total + epsilon))


"""
    A class implemented the U-net model by Keras.
"""


class IUNet:
    def __init__(self, nx, ny):
        # Construct the network architecture here. Expose only the keras.model variable as a member variable

        ### STANDARD U-NET
        # contracting path

        # down-level 1
        input_layer = layers.Input((ny, nx, 3))
        conv_1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_layer)
        conv_1 = tf.keras.layers.BatchNormalization()(conv_1)
        conv_1 = tf.keras.layers.Dropout(0.2)(conv_1)
        conv_2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(conv_1)
        conv_2 = tf.keras.layers.BatchNormalization()(conv_2)
        mPool_1 = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(conv_2)

        # down-level 2
        conv_3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(mPool_1)
        conv_3 = tf.keras.layers.BatchNormalization()(conv_3)
        conv_3 = tf.keras.layers.Dropout(0.2)(conv_3)
        conv_4 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv_3)
        conv_4 = tf.keras.layers.BatchNormalization()(conv_4)
        mPool_2 = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(conv_4)

        # down-level 3
        conv_5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(mPool_2)
        conv_5 = tf.keras.layers.BatchNormalization()(conv_5)
        conv_5 = tf.keras.layers.Dropout(0.25)(conv_5)
        conv_6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv_5)
        conv_6 = tf.keras.layers.BatchNormalization()(conv_6)
        mPool_3 = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(conv_6)

        # down-level 4
        conv_7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(mPool_3)
        conv_7 = tf.keras.layers.BatchNormalization()(conv_7)
        conv_7 = tf.keras.layers.Dropout(0.2)(conv_7)
        conv_8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv_7)
        conv_8 = tf.keras.layers.BatchNormalization()(conv_8)
        mPool_4 = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(conv_8)

        # bottom-level
        conv_9 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(mPool_4)
        conv_9 = tf.keras.layers.BatchNormalization()(conv_9)
        conv_9 = tf.keras.layers.Dropout(0.35)(conv_9)
        conv_10 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv_9)
        conv_10 = tf.keras.layers.BatchNormalization()(conv_10)

        # expanding path
        # up-level 4
        uconv_1 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv_10)
        cat_1 = layers.Concatenate()([uconv_1, conv_8])
        conv_11 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(cat_1)
        conv_11 = tf.keras.layers.BatchNormalization()(conv_11)
        conv_11 = tf.keras.layers.Dropout(0.2)(conv_11)
        conv_12 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv_11)
        conv_12 = tf.keras.layers.BatchNormalization()(conv_12)

        # up-level 3
        uconv_2 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv_12)
        cat_2 = layers.Concatenate()([uconv_2, conv_6])
        conv_13 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(cat_2)
        conv_13 = tf.keras.layers.BatchNormalization()(conv_13)
        conv_13 = tf.keras.layers.Dropout(0.25)(conv_13)
        conv_14 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv_13)
        conv_14 = tf.keras.layers.BatchNormalization()(conv_14)

        # up-level 2
        uconv_3 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv_14)
        cat_3 = layers.Concatenate()([uconv_3, conv_4])
        conv_15 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(cat_3)
        conv_15 = tf.keras.layers.BatchNormalization()(conv_15)
        conv_15 = tf.keras.layers.Dropout(0.25)(conv_15)
        conv_16 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv_15)
        conv_16 = tf.keras.layers.BatchNormalization()(conv_16)

        # up-level 1
        uconv_4 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv_16)
        cat_4 = layers.Concatenate()([uconv_4, conv_2])
        conv_17 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(cat_4)
        conv_17 = tf.keras.layers.BatchNormalization()(conv_17)
        conv_17 = tf.keras.layers.Dropout(0.25)(conv_17)
        conv_18 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(conv_17)
        conv_18 = tf.keras.layers.BatchNormalization()(conv_18)
        output = layers.Conv2D(2, (1, 1), activation='softmax', padding='same')(conv_18)
        self.iunet = tf.keras.Model(inputs=input_layer, outputs=output)

    """
        compile model and display the summary of it
    """

    def my_compile(self):
        self.iunet.compile(optimizer='adam', loss=dice_coef_loss, metrics=[dice_coef])
        print(self.iunet.summary())

    """
        fit the model on the provided training and validation inputs
    """

    def my_fit(self, tr_input, tr_mask, val_input, val_mask, batch_size, epochs, steps, callback):
        self.history = self.iunet.fit(tr_input, tr_mask, validation_data=(val_input, val_mask),
                                      batch_size=batch_size, epochs=epochs, steps_per_epoch=steps, callbacks=[callback])

    """
        save model parameter to the given path
    """

    def my_save(self, path):
        self.iunet.save(path)

    """
        Makes predictions with the trained model and returns the output segmentations.
    """

    def predict(self, data, batch_size=None):
        return self.iunet.predict(data, batch_size=batch_size, verbose=1)

    """
        keras evaluate function.
    """

    def evaluate(self, test_images, test_labels):
        return self.iunet.evaulate(test_images, test_labels, verbose=1)

    """
        load saved weights from previous training model
    """

    def load_weights(self, path):
        self.iunet.load_weights(path)
