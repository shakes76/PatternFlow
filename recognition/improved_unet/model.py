import tensorflow as tf
def context_module(input, filter):
    d0 = tf.keras.layers.BatchNormalization()(input)
    d1 = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(d0)
    d2 = tf.keras.layers.Dropout(0.3)(d1)
    d3 = tf.keras.layers.BatchNormalization()(d2)
    d4 = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(d3)
    return d4

def upsampling_module(input, filter):
    u1 = tf.keras.layers.UpSampling2D(size=(2,2))(input)
    u2 = tf.keras.layers.Conv2D(filter, (2,2), padding = 'same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(u1)
    return u2

def localiztion_module(input, filter):
    c1 = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(input)
    c2 = tf.keras.layers.Conv2D(filter, (1,1), padding = 'same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(c1)
    return c2

def unet_model(output_channels):
    inputs = tf.keras.layers.Input(shape=(256,256,1))
   
    #Encoding
    con1 = tf.keras.layers.Conv2D(16, (3,3) , padding = 'same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(inputs)    
    up1 = upsampling_module(add5, 128)

   
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