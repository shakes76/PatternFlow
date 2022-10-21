'''
Subclassed UNet model.
'''

import tensorflow as tf
from layers import My_Conv2D, Downshift, Upshift

class My_UNet(tf.keras.Model):
    def __init__(self, input_img_shape):
        '''
        Inherits from tf.keras.Model, consists of an initival MyConv2D with activation 
        passing to consecutive Downshift layers (here 5). These 'bottom out' before 
        entering a sequence of Upshift layers corresponding to the Downshifts. A context 
        aggregation pathway is attached to the Upshift sequence consisting of upsampled 
        segmentations from each Upshift layer, these being consecutively added element-
        wise before being similarly combined with the output of the Upshifts. Finally, 
        this output is passed to a segmentation layer.
        
        Args:
            input_img_shape (x, y, channels): Input image shape, must remain fixed.
        '''
        super(My_UNet, self).__init__()
        self.input_img_shape = input_img_shape
        
        self.conv2d_1 = My_Conv2D(16)
        
        self.downshift_1 = Downshift(16)
        self.downshift_2 = Downshift(32)
        self.downshift_3 = Downshift(64)
        self.downshift_4 = Downshift(128)
        self.downshift_5 = Downshift(256, bottom=True)
        
        self.upshift_1 = Upshift(128)
        self.upshift_2 = Upshift(64)
        self.upshift_3 = Upshift(32)
        self.upshift_4 = Upshift(16, top=True)
        
        self.upsampling_1 = tf.keras.layers.UpSampling2D()
        self.add_1 = tf.keras.layers.Add()
        self.upsampling_2 = tf.keras.layers.UpSampling2D()
        self.add_2 = tf.keras.layers.Add()
        
        self.output_1 = tf.keras.layers.Conv2D(4, 1, activation='softmax', padding='same')
        
    def call(self, x_0):
        '''
        Defines the forward pass.
        
        Args:
            x_0 (tf.Tensor): The input image.
            
        Returns:
            x_17 (tf.Tensor): Segmentation masks of the input image.
        '''
        x_1 = self.conv2d_1(x_0)
        
        x_2, x_2_res = self.downshift_1(x_1)
        x_3, x_3_res = self.downshift_2(x_2)
        x_4, x_4_res = self.downshift_3(x_3)
        x_5, x_5_res = self.downshift_4(x_4)
        x_6 = self.downshift_5(x_5)

        x_7, x_7_seg = self.upshift_1([x_6, x_5_res])
        x_8, x_8_seg = self.upshift_2([x_7, x_4_res])
        x_9, x_9_seg = self.upshift_3([x_8, x_3_res])
        x_10 = self.upshift_4([x_9, x_2_res])
        
        x_11 = self.upsampling_1(x_7_seg)
        x_12 = self.add_1([x_11, x_8_seg])
        x_13 = self.upsampling_1(x_12)
        x_14 = self.add_1([x_13, x_9_seg])
        x_15 = self.upsampling_2(x_14)
        x_16 = self.add_2([x_15, x_10])

        x_17 = self.output_1(x_16)
        
        return x_17
    
    def build_graph(self):
        '''
        Builds a shell of the model to implement summary and plotting functionality; 
        To emphasise the flexibility of the subclassing API, these functionalities of 
        Keras do not include native support for subclassed objects.
        
        Returns:
            tf.keras.Model: A model shell linking a defined input to the forward pass of 
            My_UNet.
        '''
        x_0 = tf.keras.Input(shape=self.input_img_shape)
        
        return tf.keras.Model(inputs=[x_0], outputs=self.call(x_0))
