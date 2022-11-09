'''
Subclassed UNet layers.
'''

import tensorflow as tf

class My_Conv2D(tf.keras.layers.Layer):
    def __init__(self, filters, activation=True, strides=(1, 1), kernel=3, \
            alpha=10^-2, initializer=tf.keras.initializers.he_uniform(), \
            regularizer=tf.keras.regularizers.l2(10^-2)):
        '''
        Inherits from tf.keras.layers.Layer, consists of a 3X3 convolution of 
        variable stride passing to an optional batch normalization and activation 
        (tf.keras.layers.LeakyReLU).
        
        Args:
            filters (int): Number of convolutional filters.
            activation (bool): Whether or not to include activation.
            strides (tuple): Stride in each dimension.
            kernel (int): Square filter side length.
            alpha (float): Gradient of tf.keras.layers.LeakyReLU below activation.
            initializer (tf.keras.initializers): Initialization algorithm, set to 
            he_uniform for ReLU based activations or glorot_uniform for others.
            kernel_regularizer (tf.keras.regularizers): Regularization algorithm, not 
            implemented here but optional for future tasks.
        '''
        super(My_Conv2D, self).__init__()
        self.conv2d_1 = tf.keras.layers.Conv2D(filters, kernel, strides=strides, \
           padding='same', kernel_initializer=initializer)
        self.activation = activation
        if self.activation:
            self.batchnorm_1 = tf.keras.layers.BatchNormalization()
            self.leakyrelu_1 = tf.keras.layers.LeakyReLU(alpha=alpha)
        
    def call(self, x):
        '''
        Defines the forward pass.
        
        Args:
            x (tf.Tensor): Image tensor, remapped if the layer is deep.
            
        Returns:
            x (tf.Tensor): Image tensor as processed by the layer.
        '''
        x = self.conv2d_1(x)
        if self.activation:
            x = self.batchnorm_1(x)
            x = self.leakyrelu_1(x)
        
        return x

class Downshift(tf.keras.layers.Layer):

    def __init__(self, filters, bottom=False, dropout=0.3):
        '''
        Inherits from tf.keras.layers.Layer, consists of a residual context module 
        passing to a stride 2X2 MyConv2D layer. The residual context module contains 
        two My_Conv2D layers, the first without activation, separated by dropout. 
        The initial tensor is then added element-wise to the output before being 
        stored as x_res and passed to the stride 2X2 convolution with twice as many 
        filters; x_res is later employed for context aggregation by Upshift. An option 
        to not compute the stride 2X2 convolution is included so that the UNet 
        architecture can effectively 'bottom out'. Combining the residual context module 
        with the stride 2X2 convolution in this way is useful for fast prototyping with 
        small networks as these converge much more quickly than larger equivalents, 
        offering a proof of concept.

        Args:
            filters (int): Number of filters to be used in each convolutional layer of 
                the residual context module.
            bottom (bool): Whether or not to 'bottom out' by not including the stride 
                2X2 convolution.
            dropout (float): Dropout probability.
        '''
        
        super(Downshift, self).__init__()
        self.bottom = bottom
        
        self.conv2d_1 = My_Conv2D(filters, activation=False)
        self.dropout_1 = tf.keras.layers.Dropout(dropout)
        self.conv2d_2 = My_Conv2D(filters)
        self.add_1 = tf.keras.layers.Add()
        self.conv2d_3 = My_Conv2D(int(filters * 2), strides=(2, 2))
        
    def call(self, x_init):
        '''
        Defines the forward pass.
        
        Args:
            x_init (tf.Tensor): Image tensor, remapped if the layer is deep.
            
        If self.bottom is not true:
            
        Returns:
            x (tf.Tensor): Image tensor as processed by the layer.
            x_res (tf.Tensor): Image tensor as processed by the layer, but 
                excluding the stride 2X2 convolution.
                
        If self.bottom is true:
        
        Returns:
            x_res (tf.Tensor): Image tensor as processed by the layer, but 
                excluding the stride 2X2 convolution.
        '''
        x = self.conv2d_1(x_init)
        x = self.dropout_1(x)
        x = self.conv2d_2(x)
        x_res = self.add_1([x, x_init])
        
        if self.bottom:
            return x_res
        else:
            x = self.conv2d_3(x)

            return x, x_res
    
class Upshift(tf.keras.layers.Layer):
    def __init__(self, filters, top=False):
        '''
        Inherits from tf.keras.layers.Layer, consists of an upsampling module passing 
        to a localization module. The upsampling module contains an upsampling layer 
        which doubles the instance of each pixel in each dimension prior to an 
        activated My_Conv2D layer which halves the feature space. Output from the 
        upsampling module then enters the localization module, where it is concatenated 
        with the x_res output from the Downshift layer of corresponding feature space 
        before a 3X3 My_Conv2D passing to a 1X1 My_Conv2D which again halves the feature 
        space. This tensor is then outputted, in addition to a segmented copy for later 
        context aggregation. As with Downshift, an option is included to bypass the 
        segmentation so that the UNet can effectively 'top out'. Again, combining the 
        upsampling and localization modules is useful for fast prototyping with small 
        networks.

        Args:
            filters (int): Number of filters to be used in each convolutional layer prior 
            to the 1X1 My_Conv2D layer.
            top (bool): Whether or not to 'top out' by not including the segmentation.
        '''
        super(Upshift, self).__init__()
        self.top = top
        
        self.upsampling_1 = tf.keras.layers.UpSampling2D()
        self.conv2d_1 = My_Conv2D(filters)
        self.concatenate_1 = tf.keras.layers.Concatenate()
        self.conv2d_2 = My_Conv2D(filters)
        self.conv2d_3 = My_Conv2D(int(filters / 2), kernel=1)
        self.seg_1 = tf.keras.layers.Conv2D(4, 1, activation='softmax', padding='same')
        
    def call(self, X):
        '''
        Defines the forward pass.
        
        Args:
            X (tf.Tensor, tf.Tensor): Image tensors x_i and x_j_res; x_i is the output from 
            the previous layer while x_j_res is the residual produced by the Downshift layer 
            of corresponding feature space.
            
        If self.top is not true:
            
        Returns:
            x (tf.Tensor): Image tensor as processed by the layer.
            x_seg (tf.Tensor): Image tensor as processed by the layer, also segmented.
                
        If self.top is true:
        
        Returns:
            x (tf.Tensor): Image tensor as processed by the layer.
        '''
        x, x_res = X
        x = self.upsampling_1(x)
        x = self.conv2d_1(x)
        x = self.concatenate_1([x, x_res])
        x = self.conv2d_2(x)
        x = self.conv2d_3(x)
        x_seg = self.seg_1(x)
        
        if self.top:
            return x_seg
        
        else:
            return x, x_seg
