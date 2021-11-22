"""
    Module to create an
    improved U-Net model.
    
    author: Richard Marron
    status: Development
"""

from tensorflow.keras import layers, Model, backend, optimizers

class ImprovedUNet():
    """Implements the Improved U-Net Model"""
    def __init__(self, input_shape: tuple, learning_rate: float=1e-3, 
                 optimizer=optimizers.Adam(1e-3), loss="binary_crossentropy",
                 leaky: float=1e-2, drop: float=3e-1):
        
        # Input must be 3-dimensional
        assert len(input_shape) == 3
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = [self.dice_function]
        
        # Leaky ReLU rate
        self.leaky = leaky
        # Drop-out rate
        self.drop = drop
    
    def model(self):
        """
        Create and return the Improved U-Net Model.
        Model based on design from the segmentation
            paper, https://arxiv.org/pdf/1802.10508v1.pdf
        """
        ################### DOWNSAMPLING ###################
        in_layer = layers.Input(shape=self.input_shape)
        
        # Initial convolution layer
        block_1 = layers.Conv2D(filters=16, kernel_size=3, padding="same")(in_layer)
        
        # First Context Module (Pre-activation residual block)
        ctx_1 = layers.BatchNormalization()(block_1)
        ctx_1 = layers.LeakyReLU(alpha=self.leaky)(ctx_1)
        ctx_1 = layers.Conv2D(filters=16, kernel_size=3, padding="same")(ctx_1)
        ctx_1 = layers.Dropout(rate=self.drop)(ctx_1)
        ctx_1 = layers.BatchNormalization()(ctx_1)
        ctx_1 = layers.LeakyReLU(alpha=self.leaky)(ctx_1)
        ctx_1 = layers.Conv2D(filters=16, kernel_size=3, padding="same")(ctx_1)
        
        # Merge/Connect path before context block to after block
        ctx_1 = layers.Add()([block_1, ctx_1])
        
        # Go down a level in the U-Net using strided convolution
        s_conv_1 = layers.Conv2D(filters=32, kernel_size=3, strides=2,
                                 padding="same", activation="relu")(ctx_1)
        
        # Second Context Module (Pre-activation residual block)
        ctx_2 = layers.BatchNormalization()(s_conv_1)
        ctx_2 = layers.LeakyReLU(alpha=self.leaky)(ctx_2)
        ctx_2 = layers.Conv2D(filters=32, kernel_size=3, padding="same")(ctx_2)
        ctx_2 = layers.Dropout(rate=self.drop)(ctx_2)
        ctx_2 = layers.BatchNormalization()(ctx_2)
        ctx_2 = layers.LeakyReLU(alpha=self.leaky)(ctx_2)
        ctx_2 = layers.Conv2D(filters=32, kernel_size=3, padding="same")(ctx_2)
        
        # Merge/Connect path before context block to after block
        ctx_2 = layers.Add()([s_conv_1, ctx_2])
        
        # Go down a level in the U-Net using strided convolution
        s_conv_2 = layers.Conv2D(filters=64, kernel_size=3, strides=2,
                                 padding="same")(ctx_2)
        
        # Third Context Module (Pre-activation residual block)
        ctx_3 = layers.BatchNormalization()(s_conv_2)
        ctx_3 = layers.LeakyReLU(alpha=self.leaky)(ctx_3)
        ctx_3 = layers.Conv2D(filters=64, kernel_size=3, padding="same")(ctx_3)
        ctx_3 = layers.Dropout(rate=self.drop)(ctx_3)
        ctx_3 = layers.BatchNormalization()(ctx_3)
        ctx_3 = layers.LeakyReLU(alpha=self.leaky)(ctx_3)
        ctx_3 = layers.Conv2D(filters=64, kernel_size=3, padding="same")(ctx_3)
        
        # Merge/Connect path before context block to after block
        ctx_3 = layers.Add()([s_conv_2, ctx_3])
        
        # Go down a level in the U-Net using strided convolution
        s_conv_3 = layers.Conv2D(filters=128, kernel_size=3, strides=2,
                                 padding="same")(ctx_3)
        
        # Fourth Context Module (Pre-activation residual block)
        ctx_4 = layers.BatchNormalization()(s_conv_3)
        ctx_4 = layers.LeakyReLU(alpha=self.leaky)(ctx_4)
        ctx_4 = layers.Conv2D(filters=128, kernel_size=3, padding="same")(ctx_4)
        ctx_4 = layers.Dropout(rate=self.drop)(ctx_4)
        ctx_4 = layers.BatchNormalization()(ctx_4)
        ctx_4 = layers.LeakyReLU(alpha=self.leaky)(ctx_4)
        ctx_4 = layers.Conv2D(filters=128, kernel_size=3, padding="same")(ctx_4)
        
        # Merge/Connect path before context block to after block
        ctx_4 = layers.Add()([s_conv_3, ctx_4])
        
        # Go down a level in the U-Net using strided convolution
        s_conv_4 = layers.Conv2D(filters=256, kernel_size=3, strides=2,
                                 padding="same")(ctx_4)
        
        # Fifth Context Module (Pre-activation residual block)
        ctx_5 = layers.BatchNormalization()(s_conv_4)
        ctx_5 = layers.LeakyReLU(alpha=self.leaky)(ctx_5)
        ctx_5 = layers.Conv2D(filters=256, kernel_size=3, padding="same")(ctx_5)
        ctx_5 = layers.Dropout(rate=self.drop)(ctx_5)
        ctx_5 = layers.BatchNormalization()(ctx_5)
        ctx_5 = layers.LeakyReLU(alpha=self.leaky)(ctx_5)
        ctx_5 = layers.Conv2D(filters=256, kernel_size=3, padding="same")(ctx_5)
        
        # Merge/Connect path before context block to after block
        ctx_5 = layers.Add()([s_conv_4, ctx_5])
        
        ################### UPSAMPLING ###################
        # First Upsampling Module 
        up_1 = layers.Conv2DTranspose(filters=128, kernel_size=3, padding="same", 
                                      strides=2)(ctx_5)
        up_1 = layers.LeakyReLU(alpha=self.leaky)(up_1)
        
        # Link across U-Net via concatenation to define first localisation module
        loc_1 = layers.concatenate([ctx_4, up_1])
        loc_1 = layers.Conv2D(filters=128, kernel_size=3, padding="same")(loc_1)
        loc_1 = layers.BatchNormalization()(loc_1)
        loc_1 = layers.LeakyReLU(alpha=self.leaky)(loc_1)

        # Second Upsampling Module 
        up_2 = layers.Conv2DTranspose(filters=64, kernel_size=3, padding="same", 
                                      strides=2)(loc_1)
        up_2 = layers.LeakyReLU(alpha=self.leaky)(up_2)
        
        # Link across U-Net via concatenation to define second localisation module
        loc_2 = layers.concatenate([ctx_3, up_2])
        loc_2 = layers.Conv2D(filters=64, kernel_size=3, padding="same")(loc_2)
        loc_2 = layers.BatchNormalization()(loc_2)
        loc_2 = layers.LeakyReLU(alpha=self.leaky)(loc_2)
        
        # Add peripheral connection for segmentation and then upscale
        seg_1 = layers.Conv2D(filters=1, kernel_size=3, padding="same", activation="sigmoid")(loc_2)
        seg_1 = layers.Conv2DTranspose(filters=1, kernel_size=3, padding="same", 
                                       strides=2)(seg_1)
        
        # Halve the number of filters
        loc_2 = layers.Conv2D(filters=32, kernel_size=1, padding="same")(loc_2)
        loc_2 = layers.LeakyReLU(alpha=self.leaky)(loc_2)
        
        # Third Upsampling Module 
        up_3 = layers.Conv2DTranspose(filters=32, kernel_size=3, padding="same", 
                                      strides=2)(loc_2)
        up_3 = layers.LeakyReLU(alpha=self.leaky)(up_3)
        
        # Link across U-Net via concatenation to define third localisation module
        loc_3 = layers.concatenate([ctx_2, up_3])
        loc_3 = layers.Conv2D(filters=32, kernel_size=3, padding="same")(loc_3)
        loc_3 = layers.BatchNormalization()(loc_3)
        loc_3 = layers.LeakyReLU(alpha=self.leaky)(loc_3)
        
        # Add peripheral connection for segmentation
        seg_2 = layers.Conv2D(filters=1, kernel_size=3, padding="same", activation="sigmoid")(loc_3)
        
        # Sum first segmentation layer with this one and upscale
        seg_partial = layers.Add()([seg_1, seg_2])
        seg_partial = layers.Conv2DTranspose(filters=1, kernel_size=3, padding="same", 
                                             strides=2)(seg_partial)
        
        # Fourth Upsampling Module 
        up_4 = layers.Conv2DTranspose(filters=16, kernel_size=3, padding="same", 
                                      strides=2)(loc_3)
        up_4 = layers.LeakyReLU(alpha=self.leaky)(up_4)
        
        # Convolution block
        block_2 = layers.concatenate([ctx_1, up_4])
        block_2 = layers.Conv2D(filters=32, kernel_size=3, padding="same")(block_2)
        
        # Segmentation layer
        seg_3 = layers.Conv2D(filters=1, kernel_size=3, padding="same", activation="sigmoid")(block_2)
        # Sum all segmentation layers
        total = layers.Add()([seg_partial, seg_3])
        
        # Define output layer
        out_layer = layers.Conv2D(filters=1, kernel_size=3, padding="same", activation="sigmoid")(total)
        
        # Return the model created from all of these layers
        return Model(in_layer, out_layer)
    
    def dice_function(self, y_true, y_pred):
        """
        Calculate the dice coefficient
            Params:
                y_true : The true values to compare
                y_pred : The predicted values to compare
            
            Return : Dice coefficient between y_true and y_pred
        """
        # Convert the milti-dim. tensors into vectors
        y_true = backend.flatten(y_true)
        y_pred = backend.flatten(y_pred)
        
        # Calculate the dice coefficient over binary vectors
        return 2.0*backend.sum(y_true*y_pred)/(backend.sum(backend.square(y_true)) + backend.sum(backend.square(y_pred)))