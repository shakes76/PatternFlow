"""
    Module to create an
    improved U-Net model.
    
    author: Richard Marron
    status: Development
"""

import tensorflow as tf

class ImprovedUNet():
    """Implements the Improved U-Net Model"""
    def __init__(self, input_shape: tuple, learning_rate: float=1e-4, 
                 optimiser=tf.keras.optimizers.Adam(1e-4), loss: str="CategoricalCrossentropy"):
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.optimizer = optimiser
        self.loss = loss
        self.metric = self.dice_function
    
    def model(self):
        """Create the Improved U-Net Model"""
        pass
    
    def dice_function(self, y_true, y_pred):
        """
        Calculate the dice coefficient
            Params:
                y_true : The true values to compare
                y_pred : The predicted values to compare
            
            Return : Dice coefficient of the images
        """
        # Convert the milti-dim. tensors into vectors
        y_true = tf.keras.flatten(y_true)
        y_pred = tf.keras.flatten(y_pred)
        
        # Calculate the dice coefficient over binary vectors
        return 2*tf.keras.sum(y_true*y_pred)/(tf.keras.sum(tf.keras.square(y_true)) + tf.keras.sum(tf.keras.square(y_pred)))
        
    