import tensorflow as tf

#Improved U-net model
def ImprovedUnet(h, w, n_channels):
    input_layer = tf.keras.layers.Input(shape = (h,w,n_channels))

    c1 = tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', padding = 'same')(input_layer)
    c1 = tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', padding = 'same')(c1)
    b1 = tf.keras.layers.BatchNormalization()(c1)
    p1 = tf.keras.layers.MaxPooling2D(pool_size = (2,2))(b1)
    
    c2 = tf.keras.layers.Conv2D(128, (3,3), activation = 'relu', padding = 'same')(p1)
    c2 = tf.keras.layers.Conv2D(128, (3,3), activation = 'relu', padding = 'same')(c2)
    b2 = tf.keras.layers.BatchNormalization()(c2)
    p2 = tf.keras.layers.MaxPooling2D(pool_size = (2,2))(b2)
    
    c3 = tf.keras.layers.Conv2D(256, (3,3), activation = 'relu', padding = 'same')(p2)
    c3 = tf.keras.layers.Conv2D(256, (3,3), activation = 'relu', padding = 'same')(c3)
    b3 = tf.keras.layers.BatchNormalization()(c3)
    p3 = tf.keras.layers.MaxPooling2D(pool_size = (2,2))(b3)
    
    c4 = tf.keras.layers.Conv2D(512, (3,3), activation = 'relu', padding = 'same')(p3)
    c4 = tf.keras.layers.Conv2D(512, (3,3), activation = 'relu', padding = 'same')(c4)
    b4 = tf.keras.layers.BatchNormalization()(c4)
    p4 = tf.keras.layers.MaxPooling2D(pool_size = (2,2))(b4)
    
    c5 = tf.keras.layers.Conv2D(1024, (3,3), activation = 'relu', padding = 'same')(p4)
    c5 = tf.keras.layers.Conv2D(1024, (3,3), activation = 'relu', padding = 'same')(c5)
    b5 = tf.keras.layers.BatchNormalization()(c5)
    
    crop1 = tf.keras.layers.Conv2DTranspose(512, (3,3), strides = (2,2), padding = 'same')(b5)
    conc1 = tf.keras.layers.concatenate([crop1, c4])
    c6 = tf.keras.layers.Conv2D(512, (3,3), activation = 'relu', padding = 'same')(conc1)
    c6 = tf.keras.layers.Conv2D(512, (3,3), activation = 'relu', padding = 'same')(c6)
    b6 = tf.keras.layers.BatchNormalization()(c6)
    
    crop2 = tf.keras.layers.Conv2DTranspose(256, (3,3), strides = (2,2), padding = 'same')(b6)
    conc2 = tf.keras.layers.concatenate([crop2, c3])
    c7 = tf.keras.layers.Conv2D(256, (3,3), activation = 'relu', padding = 'same')(conc2)
    c7 = tf.keras.layers.Conv2D(256, (3,3), activation = 'relu', padding = 'same')(c7)
    b7 = tf.keras.layers.BatchNormalization()(c7)
    
    crop3 = tf.keras.layers.Conv2DTranspose(128, (3,3), strides = (2,2), padding = 'same')(b7)
    conc3 = tf.keras.layers.concatenate([crop3, c2])
    c8 = tf.keras.layers.Conv2D(128, (3,3), activation = 'relu', padding = 'same')(conc3)
    c8 = tf.keras.layers.Conv2D(128, (3,3), activation = 'relu', padding = 'same')(c8)
    b8 = tf.keras.layers.BatchNormalization()(c8)
    
    crop4 = tf.keras.layers.Conv2DTranspose(64, (3,3), strides = (2,2), padding = 'same')(b8)
    conc4 = tf.keras.layers.concatenate([crop4, c1])
    c9 = tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', padding = 'same')(conc4)
    c9 = tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', padding = 'same')(c9)
    b9 = tf.keras.layers.BatchNormalization()(c9)
    
    output_layer = tf.keras.layers.Conv2D(4, (1,1), activation = 'softmax')(b9)
    
    return (input_layer, output_layer)