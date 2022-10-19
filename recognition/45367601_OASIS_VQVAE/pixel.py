import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import layers


# The first layer is the PixelCNN layer. This layer simply
# builds on the 2D convolutional layer, but includes masking.
class PixelConvLayer(layers.Layer):
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


# Next, we build our residual block layer.
# This is just a normal residual block, but based on the PixelConvLayer.
class ResidualBlock(layers.Layer):
    def __init__(self, filters, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(
            filters=filters, kernel_size=1, activation="relu"
        )
        self.pixel_conv = PixelConvLayer(
            mask_type="B",
            filters=filters // 2,
            kernel_size=3,
            activation="relu",
            padding="same",
            )
        self.conv2 = layers.Conv2D(filters=filters, kernel_size=1, activation="relu")
        self.norm1 = layers.BatchNormalization()
        self.norm2 = layers.BatchNormalization()
        self.norm3 = layers.BatchNormalization()


    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.norm1(x)
        x = self.pixel_conv(x)
        x = self.norm2(x)
        x = self.conv2(x)
        x = self.norm3(x)
        return layers.add([inputs, x])

class PixelCNN(keras.Model):
    
    def __init__(self, input_shape, vqvae_trainer, 
                num_pixelcnn_layers, num_residual_blocks, **kwargs):
        super(PixelCNN, self).__init__(**kwargs)

        self.pixelcnn_input_shape = input_shape
        self.num_layers = num_pixelcnn_layers
        self.num_residual_blocks = num_residual_blocks
        self.vqvae_trainer = vqvae_trainer
        self.model = self.build_cnn()


    def build_cnn(self):
        pixelcnn_inputs = keras.Input(shape=self.pixelcnn_input_shape, dtype=tf.int32)
        ohe = tf.one_hot(pixelcnn_inputs, self.vqvae_trainer.num_embeddings)
        x = PixelConvLayer(mask_type="A", filters=128, kernel_size=7, 
                            activation="relu", padding="same")(ohe)

        for _ in range(self.num_residual_blocks):
            x = ResidualBlock(filters=128)(x)

        for _ in range(self.num_layers):
            x = layers.Dropout(0.3)(x)
            x = PixelConvLayer(
                mask_type="B",
                filters=128,
                kernel_size=1,
                strides=1,
                activation="relu",
                padding="valid",
                )(x)
            x = layers.BatchNormalization()(x)
            

        out = layers.Conv2D(
            filters=self.vqvae_trainer.num_embeddings, kernel_size=1, strides=1, padding="valid")(x)

        pixel_cnn = keras.Model(pixelcnn_inputs, out, name="pixel_cnn")
        return pixel_cnn

    # def run_quantisation(self, image):
    #     '''
    #     performs quantisation to obtain the VQ representation of the supplied
    #     image dataset
    #     '''
    #     encoded_outputs = encoder.predict(image)
    #     flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
    #     codebook_indices = quantizer.get_code_indices(flat_enc_outputs)
    #     codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])
    #     return codebook_indices

    # def call(self, inputs):
    #     # input = self.run_quantisation(inputs)
    #     return super.call(input)
