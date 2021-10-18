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
        
        # Context Module (Pre-activation residual block)
        ctx_1 = layers.Conv2D(filters=16, kernel_size=3, padding="same", activation="relu")(block_1)
        ctx_1 = layers.LeakyReLU(alpha=self.leaky)(ctx_1)
        ctx_1 = layers.Dropout(rate=self.drop)(ctx_1)
        ctx_1 = layers.Conv2D(filters=16, kernel_size=3, padding="same", activation="relu")(ctx_1)
        ctx_1 = layers.LeakyReLU(alpha=self.leaky)(ctx_1)
        ctx_1 = layers.Dropout(rate=self.drop)(ctx_1)
        
        # Merge/Connect path before context block to after block
        ctx_1 = layers.Add()([block_1, ctx_1])
    
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
        
    