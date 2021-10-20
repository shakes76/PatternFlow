import os
# Suppress the INFO message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Conv2DTranspose, concatenate
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.optimizers import Adam


class SegModel:
    def __init__(self, input_shape, random_seed, model="Unet"):
        """
        Initialize SegModel class which constructs the segmentation model

        Parameters
        ----------
        input_shape : tuple
          The input shape of the model
        random_seed : integer
          The random seed to decide the random weights in the model
        model : string
          The parameter to decide which model to use. "Unet" is the baseline Unet model.
        """
        if model == "Unet":
            self.model = self.Unet(input_shape, random_seed)
        else:
            raise ValueError("Model doesn't exist!")

    def Unet(self, input_shape, random_seed):
        """
        Function to construct the baseline Unet model

        Parameters
        ----------
        input_shape : tuple
          The input shape of the model
        random_seed : integer
          The random seed to decide the random weights in the model

        Returns
        -------
        model : Keras model class
          The baseline Unet model itself
        """
        # Initialize the random seed using he_normal
        he_norm = he_normal(seed=random_seed)

        # Left side
        inputs = Input(input_shape)
        conv1 = Conv2D(64, 3, strides=1, padding='same', activation='relu', kernel_initializer=he_norm)(inputs)
        conv1 = Conv2D(64, 3, strides=1, padding='same', activation='relu', kernel_initializer=he_norm)(conv1)
        pool1 = MaxPool2D()(conv1)

        conv2 = Conv2D(128, 3, strides=1, padding='same', activation='relu', kernel_initializer=he_norm)(pool1)
        conv2 = Conv2D(128, 3, strides=1, padding='same', activation='relu', kernel_initializer=he_norm)(conv2)
        pool2 = MaxPool2D()(conv2)

        conv3 = Conv2D(256, 3, strides=1, padding='same', activation='relu', kernel_initializer=he_norm)(pool2)
        conv3 = Conv2D(256, 3, strides=1, padding='same', activation='relu', kernel_initializer=he_norm)(conv3)
        pool3 = MaxPool2D()(conv3)

        conv4 = Conv2D(512, 3, strides=1, padding='same', activation='relu', kernel_initializer=he_norm)(pool3)
        conv4 = Conv2D(512, 3, strides=1, padding='same', activation='relu', kernel_initializer=he_norm)(conv4)
        pool4 = MaxPool2D()(conv4)

        # bridge
        conv5 = Conv2D(1024, 3, strides=1, padding='same', activation='relu', kernel_initializer=he_norm)(pool4)
        conv5 = Conv2D(1024, 3, strides=1, padding='same', activation='relu', kernel_initializer=he_norm)(conv5)

        # Right side
        up6 = Conv2DTranspose(512, 3, strides=2, padding='same', kernel_initializer=he_norm)(conv5)
        concat6 = concatenate([conv4, up6], axis=3)
        conv6 = Conv2D(512, 3, strides=1, padding='same', activation='relu', kernel_initializer=he_norm)(concat6)
        conv6 = Conv2D(512, 3, strides=1, padding='same', activation='relu', kernel_initializer=he_norm)(conv6)

        up7 = Conv2DTranspose(256, 3, strides=2, padding='same', kernel_initializer=he_norm)(conv6)
        concat7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, strides=1, padding='same', activation='relu', kernel_initializer=he_norm)(concat7)
        conv7 = Conv2D(256, 3, strides=1, padding='same', activation='relu', kernel_initializer=he_norm)(conv7)

        up8 = Conv2DTranspose(128, 3, strides=2, padding='same', kernel_initializer=he_norm)(conv7)
        concat8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, strides=1, padding='same', activation='relu', kernel_initializer=he_norm)(concat8)
        conv8 = Conv2D(128, 3, strides=1, padding='same', activation='relu', kernel_initializer=he_norm)(conv8)

        up9 = Conv2DTranspose(64, 3, strides=2, padding='same', kernel_initializer=he_norm)(conv8)
        concat9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, 3, strides=1, padding='same', activation='relu', kernel_initializer=he_norm)(concat9)
        conv9 = Conv2D(64, 3, strides=1, padding='same', activation='relu', kernel_initializer=he_norm)(conv9)

        conv10 = Conv2D(2, 3, strides=1, padding='same', activation='relu', kernel_initializer=he_norm)(conv9)
        outputs = Conv2D(1, 1, activation='sigmoid', kernel_initializer=he_norm)(conv10)

        model = Model(inputs=inputs, outputs=outputs)

        return model

    def summary(self):
        """
        Print the summary of the current segmentation model in SegModel class
        """
        self.model.summary()

    def train(self, X_train, X_val, y_train, y_val, optimizer, lr, loss, metrics, batch_size, epochs):
        """
        Function to train the current segmentation model in SegModel class

        Parameters
        ----------
        X_train : float32 numpy array
          The train set of data type float32 numpy array of the preprocessed images
        X_val : float32 numpy array
          The validation set of data type float32 numpy array of the preprocessed images
        y_train : float32 numpy array
          The train set of data type float32 numpy array of the preprocessed masks
        y_val : float32 numpy array
          The validation set of data type float32 numpy array of the preprocessed masks
        optimizer : string
          The parameter to decide which optimizer to use. "adam" is using the Adam optimizer.
        lr : float
          The parameter of the learning rate
        loss : function
          The loss function used for training
        metrics : list
          A list of metric functions to evaluate train and validation when training
        batch_size : integer
          Number of samples to take in to calculate then update weights
        epochs : integer
          Number to decide how many iterations of the model is train over the whole train data set
        """
        if optimizer == 'adam':
            opt = Adam(learning_rate=lr)
        else:
            raise ValueError("Optimizer doesn't exists!")

        self.model.compile(optimizer=opt, loss=loss, metrics=metrics)
        self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, validation_data=(X_val, y_val))

    def predict(self, X_test, batch_size):
        """
        Function to predict masks on images using the current segmentation model in SegModel class

        Parameters
        ----------
        X_test : float32 numpy array
          The test set of data type float32 numpy array of the preprocessed images
        batch_size : integer
          Number of samples to take to predict at once

        Returns
        -------
        y_pred : float32 tensor
          Returns all the predicted masks
        """
        y_pred = self.model.predict(X_test, batch_size=batch_size)

        return y_pred
