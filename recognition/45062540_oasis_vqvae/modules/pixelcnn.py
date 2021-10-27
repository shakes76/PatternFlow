import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class MaskedConvLayer(layers.Layer):
    """
    Create PixelCNN layer with masks.
    """
    def __init__(self, mask_type, **kwargs):
        """
        Create a PixelCNN layer with masks.
        
        Params:
            mask_type: an alphabet character indicating the mask type, value = 'A' or 'B'
            **kwargs: additional keyword arguments
        """
        super(MaskedConvLayer, self).__init__()
        self.mask_type = mask_type
        self.conv = layers.Conv2D(**kwargs)

    def build(self, input_shape):
        """
        Create the variables of the layer.
        
        Params:
            input_shape(tf.TensorShape): Instance of TensorShape, or list of instances of TensorShape 
            if the layer expects a list of inputs (one instance per input).
        """
        # Build the conv2d layer to initialize kernel variables
        self.conv.build(input_shape)
        
        # Use the initialized kernel to create the mask
        kernel_shape = self.conv.kernel.get_shape()
        #set mask value of rows above to 1
        part1 = tf.ones((kernel_shape[0] // 2, kernel_shape[1], kernel_shape[2], kernel_shape[3]))        
        #set mask value of row below to 0
        part3 = tf.zeros((kernel_shape[0] // 2, kernel_shape[1], kernel_shape[2], kernel_shape[3]))

        if self.mask_type == "A":
            #set mask value of cells in the same row but before the current pixel to 1
            c1 = tf.ones((1, kernel_shape[1] // 2, kernel_shape[2], kernel_shape[3]))
            c2 = tf.zeros((1, kernel_shape[1] - kernel_shape[1] // 2, kernel_shape[2], kernel_shape[3]))
            part2 = tf.concat([c1,c2], axis = 1)
        else:
            #set mask value of cells in the same row but before the current pixel to 1
            #set the mask value of the center pixel to 1 if mask type is B
            c3 = tf.ones((1, kernel_shape[1] - kernel_shape[1] // 2, kernel_shape[2], kernel_shape[3]))
            c4 = tf.zeros((1, kernel_shape[1] // 2, kernel_shape[2], kernel_shape[3]))
            part2 = tf.concat([c3,c4], axis = 1)
                
        self.mask = tf.concat([part1,part2,part3], axis = 0)
                              
    def call(self, inputs):
        """
        Customize the forward pass behavior.
        
        Params:
            inputs(tf.Tensor): the input data
            
        Returns:
            (tf.Tensor): the output of the layer
        """
        #set the kernel based on weights of masks
        self.conv.kernel.assign(self.conv.kernel * self.mask)
        return self.conv(inputs)


class ResidualBlock(keras.layers.Layer):
    """
    Create residual block layer.
    """
    def __init__(self, filters, **kwargs):
        """
        Create a residual block layer.
        
        Params:
            filters: the size of the filter
        """
        super(ResidualBlock, self).__init__(**kwargs)
        self.conv1 = keras.layers.Conv2D(filters=filters, kernel_size=1, activation="relu")
        self.norm1 = keras.layers.BatchNormalization()
        self.pixel_conv = MaskedConvLayer(
            mask_type="B",
            filters=filters // 2,
            kernel_size=3,
            activation="relu",
            padding="same",
        )
        self.norm2 = keras.layers.BatchNormalization()
        self.conv2 = keras.layers.Conv2D(filters=filters, kernel_size=1, activation="relu")
        self.norm3 = keras.layers.BatchNormalization()


    def call(self, inputs):
        """
        Customize the forward pass behavior.
        
        Params:
            inputs(tf.Tensor): the input data
            
        Returns:
            (tf.Tensor): sum of the input data and output, has the same shape as the inputs.
        """
        x = self.conv1(inputs)
        x = self.norm1(x)
        x = self.pixel_conv(x)
        x = self.norm2(x)
        x = self.conv2(x)
        x = self.norm3(x)
        return keras.layers.add([inputs, x])

class PixelCNN(tf.keras.Model):
    """
    Create PixelCNN model.
    """
    def __init__(self, num_residual_blocks, num_pixelcnn_layers, num_embeddings, **kwargs):
        """
        Create a PixelCNN model.
    
        Params:
            num_residual_blocks: number of residual blocks
            num_pixelcnn_layers: number of pixelcnn layers with mask type B
            num_embeddings: the number of embeddings in the codebook
            **kwargs: additional keyword arguments
        """
        super().__init__(**kwargs)
        self.num_residual_blocks = num_residual_blocks
        self.num_pixelcnn_layers = num_pixelcnn_layers 
        self.layer1 = MaskedConvLayer(mask_type="A", filters=128, kernel_size=7, activation="relu", strides = 1,
                       padding="same")
        self.norm1 = keras.layers.BatchNormalization()

        self.layer_blocks = []
        
        for i in range(num_residual_blocks):
            self.layer_blocks.append(ResidualBlock(filters=128))
        
        self.pixel_layers = []
        self.norm_layers = []
        for i in range(num_pixelcnn_layers):
            self.pixel_layers.append(MaskedConvLayer(mask_type="B",filters=128,kernel_size=1,strides=1,
                               activation="relu",padding="valid"))
            self.norm_layers.append(keras.layers.BatchNormalization())

            
        self.outputs = keras.layers.Conv2D(filters=num_embeddings, 
                                  kernel_size=1, strides=1, padding="valid")
        
    def call(self, x):
        """
        Customize the forward pass behavior.
        
        Params:
            x(tf.Tensor): the input data
        
        Returns:
            (tf.Tensor): the output of the model
        """
        x = self.layer1(x)
        x = self.norm1(x)
        for i in range(self.num_residual_blocks):
            x = self.layer_blocks[i](x)
        
        for i in range(self.num_pixelcnn_layers):
            x = self.pixel_layers[i](x)
            x = self.norm_layers[i](x)
           
        x = self.outputs(x)
        return x
