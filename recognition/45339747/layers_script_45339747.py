"""
Laterality classification of the OAI AKOA knee data set.

@author Jonathan Godbold, s4533974.

Usage of this file is strictly for The University of Queensland.
Date: 27/10/2020.

Description:
Builds a model of the OASIS OKOA dataset.
"""

# Import libraries.
import tensorflow as tf

# Print the current version.
print('TensorFlow version:', tf.__version__)

def addLayer(model, input_shape, weight_decay, n_filters, kernel_size, padding, kernel_regularizer, batch_norm, activation_func):
    """
    Adds a convolutional 2D layer to current model.
    Format returned: model to be trained.
    Paramters:
    model - model to add layer to.
    input_shape - shape of input image.
    weight_decay - learning rate.
    n_filters - number of filters in the convolutional layer.
    kernel_size - size of kernel.
    padding - type.
    kernel_regularizer - L2 or L1.
    batch_norm - true if batch is normalized, false otherwise.
    activation_func - Normally ReLu or Sigmoid activation.
    """
    model.add(Conv2D(filters=n_filters, kernel_size=kernel_size, padding=padding, kernel_regularizer=kernel_regularizer, input_shape=input_shape))
    if (batch_norm == True):
        model.add(BatchNormalization())
    model.add(Activation(activation_func))
    return model

def buildNetwork(train_images):
    """
    Builds a network given the specified parameters.
    Format returned: model to be trained.
    """
    model = Sequential()
    shape = train_images
    weight_decay = 1e-4
    k_size = (3, 3)
    reg = regularizers.l2(weight_decay)
    model = addLayer(model, shape, weight_decay, 32, k_size, "same", reg, True, 'relu')
    model = addLayer(model, shape, weight_decay, 64, k_size, "same", reg, True, 'relu')
    model = addLayer(model, shape, weight_decay, 128, k_size, "same", reg, True, 'relu')
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())
    return model

def compile_and_run(model, epochs, batch):
    """
    Compiles and runs the model.
    - Uses Adam optimizer.
    - Loss function is binary_crossentropy.
    """
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(train_images, train_images_y, epochs, validation_data=(validate_images, validate_images_y), batch_size = batch)

print("Model successfully built and tested. Application exiting...")
