"""
Module defining Improved UNet layer blocks and model class.
"""

import tensorflow as tf

### Improved UNet layer blocks ###

# Context pathway

class ContextModule(tf.keras.layers.Layer):
    def __init__(self, filters=16):
        super(ContextModule, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters, 3, activation='relu',
                                            padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters, 3, activation='relu',
                                            padding='same')
        self.drop = tf.keras.layers.Dropout(0.3)
    
    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.drop(x)

class ContextLayerBlock(tf.keras.layers.Layer):
    def __init__(self, filters=16, first=False):
        super(ContextLayerBlock, self).__init__()
        if first:
            # First context layer does not change image size
            self.conv = tf.keras.layers.Conv2D(filters, 3, activation='relu',
                                               padding='same')
        else:
            # Subsequent context layers reduce the layer size by 1/2 by using
            # stride of 2
            self.conv = tf.keras.layers.Conv2D(filters, 3, activation='relu',
                                               strides=2, padding='same')
        self.context = ContextModule(filters)
    
    def call(self, x):
        y = self.conv(x)
        z = self.context(y)
        return (y + z)

# Localisation pathway

class LocalisationModule(tf.keras.layers.Layer):
    def __init__(self, filters=16):
        super(LocalisationModule, self).__init__()
        # Originally have double the amount of filters due to
        # previous concatenation
        self.conv1 = tf.keras.layers.Conv2D(filters*2, 3, activation='relu',
                                            padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters, 1, activation='relu',
                                            padding='same')
    
    def call(self, x):
        x = self.conv1(x)
        return self.conv2(x)
    
class LocalisationLayerBlock(tf.keras.layers.Layer):
    def __init__(self, filters=16, last=False):
        super(LocalisationLayerBlock, self).__init__()
        # Upsampling Module
        self.up_samp = tf.keras.layers.UpSampling2D(2)
        self.us_conv = tf.keras.layers.Conv2D(filters, 3, activation='relu',
                                              padding='same')
        # Concatenate
        self.concat = tf.keras.layers.Concatenate()
        if last:
            # Final localisation layer does not reduce number of filters
            # before segmentation
            self.conv = tf.keras.layers.Conv2D(filters*2, 3, activation='relu',
                                               padding='same')
        else:
            # Localisation Module
            self.conv = LocalisationModule(filters)
        # Segmentation
        self.seg = tf.keras.layers.Conv2D(2, 1, activation='softmax',
                                          padding='same')
    
    def call(self, x, y):
        x = self.up_samp(x)
        x = self.us_conv(x)
        x = self.concat([x, y])
        x = self.conv(x)
        return x, self.seg(x)
    
class SegmentationBlock(tf.keras.layers.Layer):
    def __init__(self):
        super(SegmentationBlock, self).__init__()
        self.up_samp1 = tf.keras.layers.UpSampling2D(2)
        self.up_samp2 = tf.keras.layers.UpSampling2D(2)
        self.seg = tf.keras.layers.Conv2D(2, 1, activation='softmax',
                                          padding='same')
    
    def call(self, s1, s2, s3):
        # Upsample and element-wise sum all segmentation layers to produced
        # final segmentation result
        x = self.up_samp1(s1)
        x = x + s2
        x = self.up_samp2(x)
        x = x + s3
        return self.seg(x)

    
### Improved UNet model class ###
    
class ImprovedUNetModel(tf.keras.Model):
    def __init__(self, filters=16):
        super(ImprovedUNetModel, self).__init__()
        # Five context layers, number of filters doubling each layer
        self.down1 = ContextLayerBlock(filters, True)
        self.down2 = ContextLayerBlock(filters*2)
        self.down3 = ContextLayerBlock(filters*(2**2))
        self.down4 = ContextLayerBlock(filters*(2**3))
        self.bottom = ContextLayerBlock(filters*(2**4))

        # Four localisation layers, number of filters halving each layer
        self.up1 = LocalisationLayerBlock(filters*(2**3))
        self.up2 = LocalisationLayerBlock(filters*(2**2))
        self.up3 = LocalisationLayerBlock(filters*2)
        self.up4 = LocalisationLayerBlock(filters, True)
        # Final layer to produce segmentation
        self.seg = SegmentationBlock()
        
    def call(self, x):
        # Save first 4 context layers for concatenation
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        x = self.bottom(d4)
        # Ignore segmentation result of lowest localisation layer
        x, ignore = self.up1(x, d4)
        # Save final 3 segmentation results for final layer
        x, s1 = self.up2(x, d3)
        x, s2 = self.up3(x, d2)
        x, s3 = self.up4(x, d1)
        return self.seg(s1, s2, s3)
