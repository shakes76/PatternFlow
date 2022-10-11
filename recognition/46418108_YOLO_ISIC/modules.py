import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, InputLayer, Dropout, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
import matplotlib 

class YOLO():
    def __init__(self, height, width):
        self._height = height
        self._width = width
        self._model = self.model()

    def model(self):

        # First layer
        model = Sequential()
        model.add(Conv2D(64, (7,7), strides=(2,2),input_shape=(self._width, self._height, 3)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D((2, 2), strides=(2,2)))

        # Second layer
        model.add(Conv2D(192, (3,3), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D((2, 2), strides=(2,2)))

        # Third layer
        model.add(Conv2D(128, (1,1), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))

        model.add(Conv2D(256, (3,3), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))

        
        model.add(Conv2D(256, (1,1), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))

        model.add(Conv2D(512, (3,3), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D((2, 2), strides=(2,2)))

        # Fourth layer
        model.add(Conv2D(256, (1,1), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Conv2D(512, (3,3), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))

        model.add(Conv2D(256, (1,1), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Conv2D(512, (3,3), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))

        model.add(Conv2D(256, (1,1), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Conv2D(512, (3,3), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))


        model.add(Conv2D(256, (1,1), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Conv2D(512, (3,3), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))

        model.add(Conv2D(512, (1,1), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))

        model.add(Conv2D(1024, (3,3), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D((2, 2), strides=(2,2)))

        # Fifth layer
        model.add(Conv2D(512, (1,1), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Conv2D(1024, (3,3), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))

        model.add(Conv2D(512, (1,1), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Conv2D(1024, (3,3), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))

        model.add(Conv2D(1024, (3,3), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))

        model.add(Conv2D(1024, (3,3), strides=(2,2), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))

        # Sixth layer
        model.add(Conv2D(1024, (3,3), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Conv2D(1024, (3,3), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))

        # Output layer
        model.add(Flatten())
        model.add(Dense(4096))
        model.add(Dense(1470))
        model.add(Reshape(target_shape=(7,7,30)))
        model.summary()

        return model

    def loss_function(self):
      pass
    

    def compileModel(self):
      pass

    def modelPredict(self, data):
      self._model.predict(data)

    def run(self, train_data, validation_data, epochs):
      self._model.fit(train_data, validation_data=validation_data, epochs=epochs)