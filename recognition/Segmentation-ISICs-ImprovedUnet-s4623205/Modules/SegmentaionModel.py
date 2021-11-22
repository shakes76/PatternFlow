"""
Unet based models to perform on segmentation tasks

@author: Jeng-Chung Lien
@student id: 46232050
@email: jengchung.lien@uqconnect.edu.au
"""
import os
# Suppress the INFO message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from tensorflow import pow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Conv2DTranspose, concatenate, LeakyReLU, Dropout, Add, UpSampling2D
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow_addons.layers import InstanceNormalization
from tensorflow_addons.optimizers import AdamW


class SegModel:
    def __init__(self, input_shape, random_seed, model="Improved_Unet"):
        """
        Initialize SegModel class which constructs the segmentation model

        Parameters
        ----------
        input_shape : tuple
          The input shape of the model
        random_seed : integer
          The random seed to decide the random weights in the model
        model : string
          The parameter to decide which model to use.
          "Unet" is the baseline Unet model.
          "Improved_Unet" is the improved version of Unet.

        References
        ----------
        "Unet", https://arxiv.org/abs/1505.04597
        "Improved_Unet", https://arxiv.org/abs/1802.10508v1
        """
        if model == "Unet":
            self.model = self.Unet(input_shape, random_seed)
        elif model == "Improved_Unet":
            self.model = self.Improved_Unet(input_shape, random_seed)
        else:
            raise ValueError("Model doesn't exist!")

        self.history = None

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

    def context_module(self, filters, inputs, kernel_initializer):
        """
        Function of the context module in the Improved Unet model.
        Layers in the order of 3x3Conv, Instance Normalization, Leaky Relu, Dropout, 3x3Conv, Instance Normalization, Leaky Relu.

        Parameters
        ----------
        filters : integer
          Number of the filters of the Convolution layer
        inputs : keras layer
          The input of the context module expected from another layer's output
        kernel_initializer : keras initializers
          Random distribution to decide the random weights in the model

        Returns
        -------
        context : keras layer
          The output of the context module
        """
        context = Conv2D(filters, 3, strides=1, padding='same', kernel_initializer=kernel_initializer)(inputs)
        context = InstanceNormalization()(context)
        context = LeakyReLU(alpha=0.01)(context)
        context = Dropout(0.3)(context)
        context = Conv2D(filters, 3, strides=1, padding='same', kernel_initializer=kernel_initializer)(context)
        context = InstanceNormalization()(context)
        context = LeakyReLU(alpha=0.01)(context)

        return context

    def localization_module(self, filters, inputs, kernel_initializer):
        """
        Function of the localization module in the Improved Unet model.
        Layers in the order of 3x3Conv, Instance Normalization, Leaky Relu, 1x1Conv, Instance Normalization, Leaky Relu.

        Parameters
        ----------
        filters : integer
          Number of the filters of the Convolution layer
        inputs : keras layer
          The input of the localization module expected from another layer's output
        kernel_initializer : keras initializers
          Random distribution to decide the random weights in the model

        Returns
        -------
        local : keras layer
          The output of the localization module
        """
        local = Conv2D(filters, 3, strides=1, padding='same', kernel_initializer=kernel_initializer)(inputs)
        local = InstanceNormalization()(local)
        local = LeakyReLU(alpha=0.01)(local)
        local = Conv2D(filters, 1, strides=1, padding='same', kernel_initializer=kernel_initializer)(local)
        local = InstanceNormalization()(local)
        local = LeakyReLU(alpha=0.01)(local)

        return local

    def upsampling_module(self, filters, inputs, kernel_initializer):
        """
        Function of the upsampling module in the Improved Unet model.
        Layers in the order of size(2, 2) Upsampling, 3x3Conv, Instance Normalization, Leaky Relu.

        Parameters
        ----------
        filters : integer
          Number of the filters of the Convolution layer
        inputs : keras layer
          The input of the upsampling module expected from another layer's output
        kernel_initializer : keras initializers
          Random distribution to decide the random weights in the model

        Returns
        -------
        up : keras layer
          The output of the upsampling module
        """
        up = UpSampling2D(2)(inputs)
        up = Conv2D(filters, 3, strides=1, padding='same', kernel_initializer=kernel_initializer)(up)
        up = InstanceNormalization()(up)
        up = LeakyReLU(alpha=0.01)(up)

        return up

    def Improved_Unet(self, input_shape, random_seed):
        """
        Function to construct the Improved Unet model

        Parameters
        ----------
        input_shape : tuple
          The input shape of the model
        random_seed : integer
          The random seed to decide the random weights in the model

        Returns
        -------
        model : Keras model class
          The Improved Unet model itself
        """
        # Initialize the random seed using he_normal
        he_norm = he_normal(seed=random_seed)

        # Left side
        inputs = Input(input_shape)
        conv1 = Conv2D(16, 3, strides=1, padding='same', kernel_initializer=he_norm)(inputs)
        conv1 = InstanceNormalization()(conv1)
        conv1 = LeakyReLU(alpha=0.01)(conv1)
        context1 = self.context_module(16, inputs=conv1, kernel_initializer=he_norm)
        add1 = Add()([conv1, context1])

        conv2 = Conv2D(32, 3, strides=2, padding='same', kernel_initializer=he_norm)(add1)
        conv2 = InstanceNormalization()(conv2)
        conv2 = LeakyReLU(alpha=0.01)(conv2)
        context2 = self.context_module(32, inputs=conv2, kernel_initializer=he_norm)
        add2 = Add()([conv2, context2])

        conv3 = Conv2D(64, 3, strides=2, padding='same', kernel_initializer=he_norm)(add2)
        conv3 = InstanceNormalization()(conv3)
        conv3 = LeakyReLU(alpha=0.01)(conv3)
        context3 = self.context_module(64, inputs=conv3, kernel_initializer=he_norm)
        add3 = Add()([conv3, context3])

        conv4 = Conv2D(128, 3, strides=2, padding='same', kernel_initializer=he_norm)(add3)
        conv4 = InstanceNormalization()(conv4)
        conv4 = LeakyReLU(alpha=0.01)(conv4)
        context4 = self.context_module(128, inputs=conv4, kernel_initializer=he_norm)
        add4 = Add()([conv4, context4])

        conv5 = Conv2D(256, 3, strides=2, padding='same', kernel_initializer=he_norm)(add4)
        conv5 = InstanceNormalization()(conv5)
        conv5 = LeakyReLU(alpha=0.01)(conv5)
        context5 = self.context_module(256, inputs=conv5, kernel_initializer=he_norm)
        add5 = Add()([conv5, context5])

        # Right side
        up6 = self.upsampling_module(128, inputs=add5, kernel_initializer=he_norm)
        concat6 = concatenate([add4, up6], axis=3)
        local6 = self.localization_module(128, inputs=concat6, kernel_initializer=he_norm)

        up7 = self.upsampling_module(64, inputs=local6, kernel_initializer=he_norm)
        concat7 = concatenate([add3, up7], axis=3)
        local7 = self.localization_module(64, inputs=concat7, kernel_initializer=he_norm)

        up8 = self.upsampling_module(32, inputs=local7, kernel_initializer=he_norm)
        concat8 = concatenate([add2, up8], axis=3)
        local8 = self.localization_module(32, inputs=concat8, kernel_initializer=he_norm)

        up9 = self.upsampling_module(16, inputs=local8, kernel_initializer=he_norm)
        concat9 = concatenate([add1, up9], axis=3)
        conv9 = Conv2D(32, 3, strides=1, padding='same', kernel_initializer=he_norm)(concat9)
        conv9 = InstanceNormalization()(conv9)
        conv9 = LeakyReLU(alpha=0.01)(conv9)

        # Segmentation layers
        seg7 = Conv2D(1, 1, strides=1, padding='same', kernel_initializer=he_norm)(local7)
        seg7 = InstanceNormalization()(seg7)
        seg7 = LeakyReLU(alpha=0.01)(seg7)
        upscale7 = UpSampling2D(2)(seg7)
        seg8 = Conv2D(1, 1, strides=1, padding='same', kernel_initializer=he_norm)(local8)
        seg8 = InstanceNormalization()(seg8)
        seg8 = LeakyReLU(alpha=0.01)(seg8)
        add8 = Add()([upscale7, seg8])
        upscale8 = UpSampling2D(2)(add8)
        seg9 = Conv2D(1, 1, strides=1, padding='same', kernel_initializer=he_norm)(conv9)
        seg9 = InstanceNormalization()(seg9)
        seg9 = LeakyReLU(alpha=0.01)(seg9)
        add9 = Add()([upscale8, seg9])
        outputs = Conv2D(1, 1, activation='sigmoid', kernel_initializer=he_norm)(add9)

        model = Model(inputs=inputs, outputs=outputs)

        return model

    def summary(self):
        """
        Print the summary of the current segmentation model in SegModel class
        """
        self.model.summary()

    def train(self, X_train, X_val, y_train, y_val, optimizer, lr, loss, metrics, batch_size, epochs, lr_decay=False, decay_rate=0.985, weight_decay=0.00001, save_model=False, save_path=None, monitor='loss', mode='min'):
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
        lr_decay : bool
          Decide whether to use learning rate decay or not
        decay_rate : float
          The decay rate of the learning rate each epoch. Is used if lr_decay is true.
        weight_decay : float
          Decoupled weight decay parameter when using "adamW" optimizer
        save_model : bool
          Set True to enable saving the best model weights
        save_path : string
          If save_model is True. Path to save the best model weights
        monitor : string
          If save_model is True. Metric to decide the best model weights
        mode : string
          If save_model is True. Decide the 'min' or 'max' of metric value to decide the best model weights

        References
        ----------
        "adamW", https://arxiv.org/abs/1711.05101
        """
        # The learning rate epoch decay
        if lr_decay:
            length = len(X_train)
            steps = length / batch_size
            boundaries = []
            values = []
            for i in range(epochs):
                boundaries.append(int(steps * (i+1)))
                values.append(lr * pow(decay_rate, i))
            boundaries.pop()

            lr = PiecewiseConstantDecay(boundaries=boundaries, values=values)

        # Apply optimizer
        if optimizer == 'adam':
            opt = Adam(learning_rate=lr)
        elif optimizer == 'adamW':
            opt = AdamW(weight_decay=weight_decay, learning_rate=lr)
        else:
            raise ValueError("Optimizer doesn't exists!")

        # Compile model
        self.model.compile(optimizer=opt, loss=loss, metrics=metrics)
        # Train model and record training history
        if save_model:
            # Add save best model weights checkpoint
            model_checkpoint_callback = ModelCheckpoint(
                filepath=save_path,
                save_weights_only=True,
                monitor=monitor,
                mode=mode,
                save_best_only=True)
            history = self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, validation_data=(X_val, y_val), callbacks=[model_checkpoint_callback])
        else:
            history = self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, validation_data=(X_val, y_val))
        self.history = history.history

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

    def load_weights(self, load_path):
        """
        Function to load weights to model

        Parameters
        ----------
        load_path : string
          Path to load the model weights
        """
        self.model.load_weights(load_path)