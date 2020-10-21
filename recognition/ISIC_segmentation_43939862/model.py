import tensorflow as tf

#Improved U-net model
def ImprovedUnet(h, w, n_channels):
    input_layer = tf.keras.layers.Input(shape = (h,w,n_channels))
    
    #Block 1
    b1_ds = tf.keras.layers.Conv2D(16, (3,3), padding = 'same')(input_layer)
    b1_res = b1_ds
    b1_m_c1 = tf.keras.layers.Conv2D(16, (3,3), padding = 'same')(b1_ds)
    b1_m_c1n = tf.keras.layers.BatchNormalization()(b1_m_c1)
    b1_m_c1a = tf.keras.layers.LeakyReLU(alpha=0.01)(b1_m_c1n)
    b1_m_do = tf.keras.layers.Dropout(rate=0.3)(b1_m_c1a)
    b1_m_c2 = tf.keras.layers.Conv2D(16, (3,3), padding = 'same')(b1_m_do)
    b1_m_c2n = tf.keras.layers.BatchNormalization()(b1_m_c2)
    b1_m_c2a = tf.keras.layers.LeakyReLU(alpha=0.01)(b1_m_c2n)
    b1_out = b1_m_c2a + b1_res
    
    #Block 2
    b2_ds = tf.keras.layers.Conv2D(32, (3,3), strides = (2,2), padding = 'same')(b1_out)
    b2_res = b2_ds
    b2_m_c1 = tf.keras.layers.Conv2D(32, (3,3), padding = 'same')(b2_ds)
    b2_m_c1n = tf.keras.layers.BatchNormalization()(b2_m_c1)
    b2_m_c1a = tf.keras.layers.LeakyReLU(alpha=0.01)(b2_m_c1n)
    b2_m_do = tf.keras.layers.Dropout(rate=0.3)(b2_m_c1a)
    b2_m_c2 = tf.keras.layers.Conv2D(32, (3,3), padding = 'same')(b2_m_do)
    b2_m_c2n = tf.keras.layers.BatchNormalization()(b2_m_c2)
    b2_m_c2a = tf.keras.layers.LeakyReLU(alpha=0.01)(b2_m_c2n)
    b2_out = b2_m_c2a + b2_res
    
    #Block 3
    b3_ds = tf.keras.layers.Conv2D(32, (3,3), strides = (2,2), padding = 'same')(b2_out)
    b3_res = b3_ds
    b3_m_c1 = tf.keras.layers.Conv2D(32, (3,3), padding = 'same')(b3_ds)
    b3_m_c1n = tf.keras.layers.BatchNormalization()(b3_m_c1)
    b3_m_c1a = tf.keras.layers.LeakyReLU(alpha=0.01)(b3_m_c1n)
    b3_m_do = tf.keras.layers.Dropout(rate=0.3)(b3_m_c1a)
    b3_m_c2 = tf.keras.layers.Conv2D(32, (3,3), padding = 'same')(b3_m_do)
    b3_m_c2n = tf.keras.layers.BatchNormalization()(b3_m_c2)
    b3_m_c2a = tf.keras.layers.LeakyReLU(alpha=0.01)(b3_m_c2n)
    b3_out = b3_m_c2a + b3_res
    
    #Block 4
    b4_ds = tf.keras.layers.Conv2D(32, (3,3), strides = (2,2), padding = 'same')(b3_out)
    b4_res = b4_ds
    b4_m_c1 = tf.keras.layers.Conv2D(32, (3,3), padding = 'same')(b4_ds)
    b4_m_c1n = tf.keras.layers.BatchNormalization()(b4_m_c1)
    b4_m_c1a = tf.keras.layers.LeakyReLU(alpha=0.01)(b4_m_c1n)
    b4_m_do = tf.keras.layers.Dropout(rate=0.3)(b4_m_c1a)
    b4_m_c2 = tf.keras.layers.Conv2D(32, (3,3), padding = 'same')(b4_m_do)
    b4_m_c2n = tf.keras.layers.BatchNormalization()(b4_m_c2)
    b4_m_c2a = tf.keras.layers.LeakyReLU(alpha=0.01)(b4_m_c2n)
    b4_out = b4_m_c2a + b4_res
    
    #Block 5
    b5_ds = tf.keras.layers.Conv2D(32, (3,3), strides = (2,2), padding = 'same')(b4_out)
    b5_res = b5_ds
    b5_m_c1 = tf.keras.layers.Conv2D(32, (3,3), padding = 'same')(b5_ds)
    b5_m_c1n = tf.keras.layers.BatchNormalization()(b5_m_c1)
    b5_m_c1a = tf.keras.layers.LeakyReLU(alpha=0.01)(b5_m_c1n)
    b5_m_do = tf.keras.layers.Dropout(rate=0.3)(b5_m_c1a)
    b5_m_c2 = tf.keras.layers.Conv2D(32, (3,3), padding = 'same')(b5_m_do)
    b5_m_c2n = tf.keras.layers.BatchNormalization()(b5_m_c2)
    b5_m_c2a = tf.keras.layers.LeakyReLU(alpha=0.01)(b5_m_c2n)
    b5_out = b5_m_c2a + b5_res
    
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
    
    output_layer = tf.keras.layers.Conv2D(2, (1,1), activation = 'softmax')(b9)
    
    return (input_layer, output_layer)