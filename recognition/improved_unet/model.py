import tensorflow as tf

"""
Retrun a Context module
With one Batch normalization -> One 2D convolution layer with LeakyRelu as a activation
->And with a 30% Dropout layer-> Batch normalization-> One 2D convolution layer with leakyRelu as a activation
"""
def context_module(input, filter):
    d0 = tf.keras.layers.BatchNormalization()(input)
    d1 = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(d0)
    d2 = tf.keras.layers.Dropout(0.3)(d1)
    d3 = tf.keras.layers.BatchNormalization()(d2)
    d4 = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(d3)
    return d4

"""
Return a Upsampling module
One Upsampling layer for 2D inputs->One 2D convolution layer with LeakyRelu as a activation
"""
def upsampling_module(input, filter):
    u1 = tf.keras.layers.UpSampling2D(size=(2,2))(input)
    u2 = tf.keras.layers.Conv2D(filter, (2,2), padding = 'same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(u1)
    return u2

"""
Return a Localiztion module
Two 2D convolution layer with LeakyRelu as a activation
"""
def localiztion_module(input, filter):
    c1 = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(input)
    c2 = tf.keras.layers.Conv2D(filter, (1,1), padding = 'same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(c1)
    return c2

"""
Will return a improved unet model
One Upsampling layer for 2D inputs->One 2D convolution layer with LeakyRelu as a activation
"""
def unet_model(output_channels):
    inputs = tf.keras.layers.Input(shape=(256,256,1))
   
    #Encoding
    con1 = tf.keras.layers.Conv2D(16, (3,3) , padding = 'same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(inputs)    
    con2 = context_module(con1, 16)
    add1 = tf.keras.layers.Add()([con1, con2])
    
    con3 = tf.keras.layers.Conv2D(32, (3,3), strides = 2 ,padding = 'same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(add1)    
    con4 = context_module(con3, 32)
    add2 = tf.keras.layers.Add()([con3, con4])
    
    con5 = tf.keras.layers.Conv2D(64, (3,3), strides = 2 ,padding = 'same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(add2)    
    con6 = context_module(con5, 64)
    add3 = tf.keras.layers.Add()([con5, con6])
    
    con7 = tf.keras.layers.Conv2D(128, (3,3), strides = 2 , padding = 'same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(add3)    
    con8 = context_module(con7, 128)
    add4 = tf.keras.layers.Add()([con7, con8])
    
    con9 = tf.keras.layers.Conv2D(256, (3,3), strides = 2 , padding = 'same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(add4)    
    con10 = context_module(con9, 256)
    add5 = tf.keras.layers.Add()([con9, con10])
  
    #Decoding
    up1 = upsampling_module(add5, 128)
    concat1 = tf.keras.layers.concatenate([up1, add4])
       
    loca1 = localiztion_module(concat1, 128)
    up2 = upsampling_module(loca1, 64)
    concat2 = tf.keras.layers.concatenate([up2, add3])
   
    loca2 = localiztion_module(concat2, 64)
    up3 = upsampling_module(loca2, 32)
    concat3 = tf.keras.layers.concatenate([up3, add2])
   
    loca3 = localiztion_module(concat3, 32)
    up4 = upsampling_module(loca3, 16)
    concat4 = tf.keras.layers.concatenate([up4, add1])

    con11 = tf.keras.layers.Conv2D(32, (3,3), padding = 'same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(concat4)
    seg1 = tf.keras.layers.Conv2D(1, (1,1), activation = tf.keras.layers.LeakyReLU(alpha=0.01), padding='same')(loca2)

    up5 = tf.keras.layers.UpSampling2D(size=(2,2))(seg1)
    seg2 = tf.keras.layers.Conv2D(1, (1,1), activation = tf.keras.layers.LeakyReLU(alpha=0.01), padding='same')(loca3)
    add6 = tf.keras.layers.Add()([up5, seg2])

    up6 = tf.keras.layers.UpSampling2D(size =(2,2))(add6)
    con12 = tf.keras.layers.Conv2D(1, (1,1), padding = 'same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(con11)
    add7 = tf.keras.layers.Add()([up6, con12])
  
    outputs = tf.keras.layers.Conv2D(4, (1,1),  activation='softmax')(add7)
    model = tf.keras.Model(inputs=inputs, outputs =outputs)
   
    return model