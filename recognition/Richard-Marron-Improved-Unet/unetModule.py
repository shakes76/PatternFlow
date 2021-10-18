"""
    Module to create an
    improved U-Net model.
    
    author: Richard Marron
    status: Development
"""

import tensorflow as tf
from tensorflow.keras import layers

class ImprovedUNet():
    """Implements the Improved U-Net Model"""
    def __init__(self, input_shape: tuple, learning_rate: float=1e-4, 
                 optimiser=tf.keras.optimizers.Adam(1e-4), loss: str="CategoricalCrossentropy",
                 leaky: float=1e-2, drop: float=3e-1):
        
        # Input must be 3-dimensional
        assert len(input_shape) == 3
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.optimizer = optimiser
        self.loss = loss
        
        # Leaky ReLU rate
        self.leaky = leaky
        # Drop-out rate
        self.drop = drop
        self.metric = self.dice_function
    
    def model(self):
        """
        Create and return the Improved U-Net Model.
        Model based on design from the segmentation
            paper, https://arxiv.org/pdf/1802.10508v1.pdf
        """
        ################### DOWNSAMPLING ###################
        in_layer = layers.Input(shape=self.input_shape)
        
        # Initial convolution layer
        block_1 = layers.Conv2D(filters=16, kernel_size=3, padding="same", activation="relu")(in_layer)
        
        # First Context Module (Pre-activation residual block)
        ctx_1 = layers.Conv2D(filters=16, kernel_size=3, padding="same", activation="relu")(block_1)
        ctx_1 = layers.LeakyReLU(alpha=self.leaky)(ctx_1)
        ctx_1 = layers.Dropout(rate=self.drop)(ctx_1)
        ctx_1 = layers.Conv2D(filters=16, kernel_size=3, padding="same", activation="relu")(ctx_1)
        ctx_1 = layers.LeakyReLU(alpha=self.leaky)(ctx_1)
        ctx_1 = layers.Dropout(rate=self.drop)(ctx_1)
        
        # Merge/Connect path before context block to after block
        ctx_1 = layers.Add()([block_1, ctx_1])
        
        # Go down a level in the U-Net using strided convolution
        s_conv_1 = layers.Conv2D(filters=32, kernel_size=3, strides=2,
                                 padding="same", activation="relu")(ctx_1)
        
        # Second Context Module (Pre-activation residual block)
        ctx_2 = layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu")(s_conv_1)
        ctx_2 = layers.LeakyReLU(alpha=self.leaky)(ctx_2)
        ctx_2 = layers.Dropout(rate=self.drop)(ctx_2)
        ctx_2 = layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu")(ctx_2)
        ctx_2 = layers.LeakyReLU(alpha=self.leaky)(ctx_2)
        ctx_2 = layers.Dropout(rate=self.drop)(ctx_2)
        
        # Merge/Connect path before context block to after block
        ctx_2 = layers.Add()([s_conv_1, ctx_2])
        
        # Go down a level in the U-Net using strided convolution
        s_conv_2 = layers.Conv2D(filters=64, kernel_size=3, strides=2,
                                 padding="same", activation="relu")(ctx_2)
        
        # Third Context Module (Pre-activation residual block)
        ctx_3 = layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu")(s_conv_2)
        ctx_3 = layers.LeakyReLU(alpha=self.leaky)(ctx_3)
        ctx_3 = layers.Dropout(rate=self.drop)(ctx_3)
        ctx_3 = layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu")(ctx_3)
        ctx_3 = layers.LeakyReLU(alpha=self.leaky)(ctx_3)
        ctx_3 = layers.Dropout(rate=self.drop)(ctx_3)
        
        # Merge/Connect path before context block to after block
        ctx_3 = layers.Add()([s_conv_2, ctx_3])
        
        # Go down a level in the U-Net using strided convolution
        s_conv_3 = layers.Conv2D(filters=128, kernel_size=3, strides=2,
                                 padding="same", activation="relu")(ctx_3)
        
        # Fourth Context Module (Pre-activation residual block)
        ctx_4 = layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu")(s_conv_3)
        ctx_4 = layers.LeakyReLU(alpha=self.leaky)(ctx_4)
        ctx_4 = layers.Dropout(rate=self.drop)(ctx_4)
        ctx_4 = layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu")(ctx_4)
        ctx_4 = layers.LeakyReLU(alpha=self.leaky)(ctx_4)
        ctx_4 = layers.Dropout(rate=self.drop)(ctx_4)
        
        # Merge/Connect path before context block to after block
        ctx_4 = layers.Add()([s_conv_3, ctx_4])
        
        # Go down a level in the U-Net using strided convolution
        s_conv_4 = layers.Conv2D(filters=256, kernel_size=3, strides=2,
                                 padding="same", activation="relu")(ctx_4)
        
        # Fifth Context Module (Pre-activation residual block)
        ctx_5 = layers.Conv2D(filters=256, kernel_size=3, padding="same", activation="relu")(s_conv_4)
        ctx_5 = layers.LeakyReLU(alpha=self.leaky)(ctx_5)
        ctx_5 = layers.Dropout(rate=self.drop)(ctx_5)
        ctx_5 = layers.Conv2D(filters=256, kernel_size=3, padding="same", activation="relu")(ctx_5)
        ctx_5 = layers.LeakyReLU(alpha=self.leaky)(ctx_5)
        ctx_5 = layers.Dropout(rate=self.drop)(ctx_5)
        
        # Merge/Connect path before context block to after block
        ctx_5 = layers.Add()([s_conv_4, ctx_5])
        
        ################### UPSAMPLING ###################
        
        
    
    def dice_function(self, y_true, y_pred):
        """
        Calculate the dice coefficient
            Params:
                y_true : The true values to compare
                y_pred : The predicted values to compare
            
            Return : Dice coefficient between y_true and y_pred
        """
        # Convert the milti-dim. tensors into vectors
        y_true = tf.keras.flatten(y_true)
        y_pred = tf.keras.flatten(y_pred)
        
        # Calculate the dice coefficient over binary vectors
        return 2*tf.keras.sum(y_true*y_pred)/(tf.keras.sum(tf.keras.square(y_true)) + tf.keras.sum(tf.keras.square(y_pred)))
        
    