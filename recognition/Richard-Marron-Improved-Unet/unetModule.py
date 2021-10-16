"""
    Module to create an
    improved U-Net model.
    
    author: Richard Marron
    status: Development
"""

import tensorflow as tf

class ImprovedUNet():
    """Implements the Improved U-Net Model"""
    def __init__(self, learning_rate=1e-4, 
                 optimiser=tf.keras.optimizers.Adam(1e-4), loss="CategoricalCrossentropy",
                 metrics=["dice_coefficient"]):
        self.learning_rate = learning_rate
        self.optimizer = optimiser
        self.metrics = metrics
    
    