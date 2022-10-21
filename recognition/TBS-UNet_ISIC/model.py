import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import tensorflow.keras.backend as K

"""
    Metric function to compute the dice score during training. This uses argmax so is not continuous and will not
    provide derivatives, but computes the true Dice score.
"""
def dice_metric(y_true, y_pred, epsilon=1e-6):
    y_true_am = K.argmax(y_true, axis=3)
    y_pred_am = K.argmax(y_pred, axis=3)

    # calculate sums over the x and y axis of each label frame in the batch
    axes = tuple(range(1, len(y_pred_am.shape)))

    # calculate the dice coefficient according to the formula (square y_true, y_pred cause it trains better)
    num = 2. * K.cast(K.sum(y_true_am * y_pred_am, axes), 'float32')
    denom = K.cast(K.sum(y_true_am + y_pred_am, axes), 'float32')

    # as this loss is being called on a batch of samples, take the average loss over the whole batch
    return K.mean(num / denom)

"""
    This is the 'softened' dice loss which provides a loss-form of the dice coefficient computed from the probabilistic
    values in the network. This allows the training algorithm to compute derivatives.
"""
def soft_dice_loss(y_true, y_pred, epsilon=1e-6):

    # calculate sums over the x and y axis of each label frame in the batch
    axes = tuple(range(1, len(y_pred.shape) - 1))

    # calculate the dice coefficient according to the formula (square y_true, y_pred cause it trains better)
    num = 2. * K.cast(K.sum(y_true * y_pred, axes), 'float32')
    denom = K.cast(K.sum(K.square(y_true) + K.square(y_pred), axes), 'float32')

    # as this loss is being called on a batch of samples, take the average loss over the whole batch
    # use the epsilon to make sure we have no divide by 0
    return 1. - K.mean((num + epsilon) / (denom + epsilon))

"""
    UNet class encapsulating the built U-Net implementation to solve ISIC problem.
    
    - Class stores the actual Keras model in the variable self.unet
    - Exposes main functionality of Keras model (compile, train, predict) with wrapper functions
    - Requires dimensions of input dataset (width/height in pixels) upon initialisation so it can build the network

"""
class UNet:
    def __init__(self, nx, ny):
        # Construct the network architecture here. Expose only the keras.model variable as a member variable

        ### STANDARD U-NET
        # contracting path

        # down-level 1
        input_layer = layers.Input((ny, nx, 3))
        conv_1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
        conv_2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv_1)
        mPool_1 = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(conv_2)

        # down-level 2
        conv_3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(mPool_1)
        conv_4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv_3)
        mPool_2 = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(conv_4)

        # down-level 3
        conv_5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(mPool_2)
        conv_6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv_5)
        mPool_3 = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(conv_6)

        # down-level 4
        conv_7 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(mPool_3)
        conv_8 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv_7)
        mPool_4 = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(conv_8)

        # bottom-level
        conv_9 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(mPool_4)
        conv_10 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(conv_9)

        # expanding path
        # up-level 4
        uconv_1 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv_10)
        cat_1 = layers.Concatenate()([uconv_1, conv_8])
        conv_11 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(cat_1)
        conv_12 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv_11)

        # up-level 3
        uconv_2 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv_12)
        cat_2 = layers.Concatenate()([uconv_2, conv_6])
        conv_13 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(cat_2)
        conv_14 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv_13)

        # up-level 2
        uconv_3 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv_14)
        cat_3 = layers.Concatenate()([uconv_3, conv_4])
        conv_15 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(cat_3)
        conv_16 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv_15)

        # up-level 1
        uconv_4 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv_16)
        cat_4 = layers.Concatenate()([uconv_4, conv_2])
        conv_17 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(cat_4)
        conv_18 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv_17)
        conv_19 = layers.Conv2D(2, (1, 1), activation='softmax', padding='same')(conv_18)

        self.unet = tf.keras.Model(inputs=input_layer, outputs=conv_19)

    """
        Function compiles and outputs a summary of the model structure
    """
    def compileModel(self):
        self.unet.compile(optimizer='adam', loss=soft_dice_loss, metrics=[dice_metric])
        print(self.unet.summary())

    """
        Trains the model on the provided training and validation data
    """
    def trainModel(self, trainData, trainLabels, validationData, validationLabels, epochs):
        self.h = self.unet.fit(trainData, trainLabels, validation_data=(validationData, validationLabels), batch_size=16, epochs=epochs)

    """
        If desired, saves the model parameters to a file for rebuild later.
    """
    def saveModel(self):
        self.unet.save("unet_model")

    """
        Makes predictions with the trained model and returns the output segmentations.
    """
    def modelPredict(self, inputData):
        return self.unet.predict(inputData)
