from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np


# PixelCNN Layer

class PixelConvLayer(layers.Layer):
    """
    The first layer is the PixelCNN layer. 
    This layer builds on the 2D convolutional layer, but includes masking.
    """
    
    def __init__(self, mask_type, **kwargs):
        super(PixelConvLayer, self).__init__()
        self.mask_type = mask_type
        self.conv = layers.Conv2D(**kwargs)

    def build(self, input_shape):
        
        # Build the conv2d layer to initialize kernel variables
        self.conv.build(input_shape)
        
        # Use the initialized kernel to create the mask
        kernel_shape = self.conv.kernel.get_shape()
        self.mask = np.zeros(shape=kernel_shape)
        self.mask[: kernel_shape[0] // 2, ...] = 1.0
        self.mask[kernel_shape[0] // 2, : kernel_shape[1] // 2, ...] = 1.0
        
        if self.mask_type == "B":
            self.mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...] = 1.0

    def call(self, inputs):
        self.conv.kernel.assign(self.conv.kernel * self.mask)
        return self.conv(inputs)


# Residual block layer.

class ResidualBlock(keras.layers.Layer):
    """
    Residual block, based on the PixelConvLayer.
    """
    
    def __init__(self, filters, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.conv1 = keras.layers.Conv2D(
            filters=filters, kernel_size=1, activation="relu"
        )
        self.pixel_conv = PixelConvLayer(
            mask_type="B",
            filters=filters // 2,
            kernel_size=3,
            activation="relu",
            padding="same",
        )
        self.conv2 = keras.layers.Conv2D(
            filters=filters, kernel_size=1, activation="relu"
        )

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pixel_conv(x)
        x = self.conv2(x)
        return keras.layers.add([inputs, x])
        

# PixelCNN Model

def get_pixelcnn(vqvae_trainer, encoded_outputs):
    """
    Builds and returns the PixelCNN model.
    """
    
    # Initialise number of PixelCNN blocks
    num_residual_blocks = 2
    num_pixelcnn_layers = 2
    pixelcnn_input_shape = encoded_outputs.shape[1:-1]
    print(f"Input shape of the PixelCNN: {pixelcnn_input_shape}")
    
    # Initialise inputs to PixelCNN
    pixelcnn_inputs = keras.Input(shape=pixelcnn_input_shape, dtype=tf.int32)
    ohe = tf.one_hot(pixelcnn_inputs, vqvae_trainer.num_embeddings)
    x = PixelConvLayer(mask_type="A", filters=128, kernel_size=7, activation="relu", padding="same")(ohe)

    # Build PixelCNN model
    for _ in range(num_residual_blocks):
        x = ResidualBlock(filters=128)(x)

    for _ in range(num_pixelcnn_layers):
        x = PixelConvLayer(
            mask_type="B",
            filters=128,
            kernel_size=1,
            strides=1,
            activation="relu",
            padding="valid",
        )(x)

    # Outputs from PixelCNN    
    out = keras.layers.Conv2D(filters=vqvae_trainer.num_embeddings, kernel_size=1, strides=1, padding="valid")(x)

    pixel_cnn = keras.Model(pixelcnn_inputs, out, name="pixel_cnn")
    
    return pixel_cnn
    
     