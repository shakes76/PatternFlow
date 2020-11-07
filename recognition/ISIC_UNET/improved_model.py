"""
Module defining Improved UNet model blocks and class.

@author: s4537175
"""

import tensorflow as tf

### Improved UNet model blocks ###

class ContextModule(tf.keras.layers.Layer):
    def __init__(self, filters=16):
        super(ContextModule, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters, 3, activation='relu', padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters, 3, activation='relu', padding='same')
        self.drop = tf.keras.layers.Dropout(0.3)
    
    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.drop(x)

class ImprovedDownBlock(tf.keras.layers.Layer):
    def __init__(self, filters=16, first=False):
        super(ImprovedDownBlock, self).__init__()
        if first:
            self.conv = tf.keras.layers.Conv2D(filters, 3, activation='relu', padding='same')
        else:
            self.conv = tf.keras.layers.Conv2D(filters, 3, activation='relu', strides=2)
        self.context = ContextModule(filters)
    
    def call(self, x):
        y = self.conv(x)
        z = self.context(y)
        return (y + z)

class LocalisationModule(tf.keras.layers.Layer):
    def __init__(self, filters=16):
        super(LocalisationModule, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters*2, 3, activation='relu', padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters, 1, activation='relu', padding='same')
    
    def call(self, x):
        x = self.conv1(x)
        return self.conv2(x)
    
class ImprovedUpBlock(tf.keras.layers.Layer):
    def __init__(self, filters=16, last=False):
        super(ImprovedUpBlock, self).__init__()
        # Upsampling Module
        self.up_samp = tf.keras.layers.UpSampling2D(2)
        self.us_conv = tf.keras.layers.Conv2D(filters, 3, activation='relu', padding='same')
        # Concatenate
        self.concat = tf.keras.layers.Concatenate()
        if last:
            self.conv = tf.keras.layers.Conv2D(filters, 3, activation='relu', padding='same')
        else:
            # Localisation Module
            self.conv = LocalisationModule(filters)
        # Segmentation
        self.seg = tf.keras.layers.Conv2D(2, 1, activation='softmax', padding='same')
    
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
        self.seg = tf.keras.layers.Conv2D(2, 1, activation='softmax', padding='same')
    
    def call(self, s1, s2, s3):
        x = self.up_samp1(s1)
        x = x + s2
        x = self.up_samp2(x)
        x = x + s3
        return self.seg(x)

    
### Improved UNet model ###
    
class ImprovedUNetModel(tf.keras.Model):
    def __init__(self, filters=16):
        super(ImprovedUNetModel, self).__init__()
        self.down_block1 = ImprovedDownBlock(filters, True)
        self.down_block2 = ImprovedDownBlock(filters*2)
        self.down_block3 = ImprovedDownBlock(filters*(2**2))
        self.down_block4 = ImprovedDownBlock(filters*(2**3))
        self.down_block5 = ImprovedDownBlock(filters*(2**4))
        
        self.up_block1 = ImprovedUpBlock(filters*(2**3))
        self.up_block2 = ImprovedUpBlock(filters*(2**2))
        self.up_block3 = ImprovedUpBlock(filters*2)
        self.up_block4 = ImprovedUpBlock(filters, True)

        self.out = SegmentationBlock()
        
    def call(self, x):
        d1 = self.down_block1(x)
        d2 = self.down_block2(d1)
        d3 = self.down_block3(d2)
        d4 = self.down_block4(d3)
        x = self.down_block5(d4)
        x, ignore = self.up_block1(x, d4)
        x, s1 = self.up_block2(x, d3)
        x, s2 = self.up_block3(x, d2)
        x, s3 = self.up_block4(x, d1)
        return self.out(s1, s2, s3)
