"""
Module defining UNet layer blocks and model class.
"""

import tensorflow as tf

### UNet layer blocks ###

class DownBlock(tf.keras.layers.Layer):
    """
    Layer block which applies 2 convolutions and a max pooling. Outputs the
    result after and before the max pooling.
    """
    def __init__(self, filters=16):
        super(DownBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters, 3, activation='relu',
                                            padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters, 3, activation='relu',
                                            padding='same')
        self.max_pool = tf.keras.layers.MaxPool2D(2)
    
    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.max_pool(x), x
        
class UpBlock(tf.keras.layers.Layer):
    """
    Layer block which applies 2 convolutions and an up sampling and outputs the
    result.
    """
    def __init__(self, filters=16):
        super(UpBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters, 3, activation='relu',
                                            padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters, 3, activation='relu',
                                            padding='same')
        self.up_samp = tf.keras.layers.UpSampling2D(2)
    
    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.up_samp(x)
    
class ConcatUpBlock(tf.keras.layers.Layer):
    """
    Layer block which concatenates two layers before appling the UpBlock layers.
    """
    def __init__(self, filters=16):
        super(ConcatUpBlock, self).__init__()
        self.concat = tf.keras.layers.Concatenate()
        self.up_block = UpBlock(filters)
    
    def call(self, x, y):
        x = self.concat([x, y])
        return self.up_block(x)   
    
class ConcatOutBlock(tf.keras.layers.Layer):
    """
    Layer block which concatenates two layers, applies 2 convolutions and
    outputs a segmentation map.
    """
    def __init__(self, filters=16):
        super(ConcatOutBlock, self).__init__()
        self.concat = tf.keras.layers.Concatenate()
        self.conv1 = tf.keras.layers.Conv2D(filters, 3, activation='relu',
                                            padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters, 3, activation='relu',
                                            padding='same')
        self.out = tf.keras.layers.Conv2D(2, 1, activation='softmax',
                                          padding='same')
    
    def call(self, x, y):
        x = self.concat([x, y])
        x = self.conv1(x)
        x = self.conv2(x)
        return self.out(x) 
    
### UNet model ###
    
class UNetModel(tf.keras.Model):
    def __init__(self, filters=16):
        super(UNetModel, self).__init__()
        # Four context / down layers, number of filters doubling each layer
        self.down_block1 = DownBlock(filters)
        self.down_block2 = DownBlock(filters*2)
        self.down_block3 = DownBlock(filters*(2**2))
        self.down_block4 = DownBlock(filters*(2**3))
        
        # Four localisation / up layers, number of filters halving each layer
        self.up_block1 = UpBlock(filters*(2**3))
        self.up_block2 = ConcatUpBlock(filters*(2**2))
        self.up_block3 = ConcatUpBlock(filters*2)
        self.up_block4 = ConcatUpBlock(filters)

        # Final layer to produce segmentation
        self.out = ConcatOutBlock(filters)
        
    def call(self, x):
        x, y1 = self.down_block1(x)
        x, y2 = self.down_block2(x)
        x, y3 = self.down_block3(x)
        x, y4 = self.down_block4(x)
        x = self.up_block1(x)
        x = self.up_block2(x, y4)
        x = self.up_block3(x, y3)
        x = self.up_block4(x, y2)
        return self.out(x, y1)
