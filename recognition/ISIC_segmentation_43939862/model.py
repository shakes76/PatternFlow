import tensorflow as tf

#Improved U-net model
def ImprovedUnet(h, w, n_channels):
    input_layer = tf.keras.layers.Input(shape = (h,w,n_channels))
    
    #Block 1 (Context)
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
    
    #Block 2 (Context)
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
    
    #Block 3 (Context)
    b3_ds = tf.keras.layers.Conv2D(64, (3,3), strides = (2,2), padding = 'same')(b2_out)
    b3_res = b3_ds
    b3_m_c1 = tf.keras.layers.Conv2D(64, (3,3), padding = 'same')(b3_ds)
    b3_m_c1n = tf.keras.layers.BatchNormalization()(b3_m_c1)
    b3_m_c1a = tf.keras.layers.LeakyReLU(alpha=0.01)(b3_m_c1n)
    b3_m_do = tf.keras.layers.Dropout(rate=0.3)(b3_m_c1a)
    b3_m_c2 = tf.keras.layers.Conv2D(64, (3,3), padding = 'same')(b3_m_do)
    b3_m_c2n = tf.keras.layers.BatchNormalization()(b3_m_c2)
    b3_m_c2a = tf.keras.layers.LeakyReLU(alpha=0.01)(b3_m_c2n)
    b3_out = b3_m_c2a + b3_res
    
    #Block 4 (Context)
    b4_ds = tf.keras.layers.Conv2D(128, (3,3), strides = (2,2), padding = 'same')(b3_out)
    b4_res = b4_ds
    b4_m_c1 = tf.keras.layers.Conv2D(128, (3,3), padding = 'same')(b4_ds)
    b4_m_c1n = tf.keras.layers.BatchNormalization()(b4_m_c1)
    b4_m_c1a = tf.keras.layers.LeakyReLU(alpha=0.01)(b4_m_c1n)
    b4_m_do = tf.keras.layers.Dropout(rate=0.3)(b4_m_c1a)
    b4_m_c2 = tf.keras.layers.Conv2D(128, (3,3), padding = 'same')(b4_m_do)
    b4_m_c2n = tf.keras.layers.BatchNormalization()(b4_m_c2)
    b4_m_c2a = tf.keras.layers.LeakyReLU(alpha=0.01)(b4_m_c2n)
    b4_out = b4_m_c2a + b4_res
    
    #Block 5 (Context)
    b5_ds = tf.keras.layers.Conv2D(256, (3,3), strides = (2,2), padding = 'same')(b4_out)
    b5_res = b5_ds
    b5_m_c1 = tf.keras.layers.Conv2D(256, (3,3), padding = 'same')(b5_ds)
    b5_m_c1n = tf.keras.layers.BatchNormalization()(b5_m_c1)
    b5_m_c1a = tf.keras.layers.LeakyReLU(alpha=0.01)(b5_m_c1n)
    b5_m_do = tf.keras.layers.Dropout(rate=0.3)(b5_m_c1a)
    b5_m_c2 = tf.keras.layers.Conv2D(256, (3,3), padding = 'same')(b5_m_do)
    b5_m_c2n = tf.keras.layers.BatchNormalization()(b5_m_c2)
    b5_m_c2a = tf.keras.layers.LeakyReLU(alpha=0.01)(b5_m_c2n)
    b5_out = b5_m_c2a + b5_res
    
    #Block 6 (Localization)
    b6_us = tf.keras.layers.Conv2DTranspose(128, (3,3), strides = (2,2), padding = 'same')(b5_out)
    b6_con = tf.keras.layers.concatenate([b6_us,b4_out])
    b6_m_c1 = tf.keras.layers.Conv2D(128, (3,3), padding = 'same')(b6_con)
    b6_m_c1n = tf.keras.layers.BatchNormalization()(b6_m_c1)
    b6_m_c1a = tf.keras.layers.LeakyReLU(alpha=0.01)(b6_m_c1n)
    b6_m_c2 = tf.keras.layers.Conv2D(128, (1,1), padding = 'same')(b6_m_c1a)
    b6_m_c2n = tf.keras.layers.BatchNormalization()(b6_m_c2)
    b6_m_c2a = tf.keras.layers.LeakyReLU(alpha=0.01)(b6_m_c2n)
    
    #Block 7 (Localization)
    b7_us = tf.keras.layers.Conv2DTranspose(64, (3,3), strides = (2,2), padding = 'same')(b6_m_c2a)
    b7_con = tf.keras.layers.concatenate([b7_us,b3_out])
    b7_m_c1 = tf.keras.layers.Conv2D(64, (3,3), padding = 'same')(b7_con)
    b7_m_c1n = tf.keras.layers.BatchNormalization()(b7_m_c1)
    b7_m_c1a = tf.keras.layers.LeakyReLU(alpha=0.01)(b7_m_c1n)
    b7_m_c2 = tf.keras.layers.Conv2D(64, (1,1), padding = 'same')(b7_m_c1a)
    b7_m_c2n = tf.keras.layers.BatchNormalization()(b7_m_c2)
    b7_m_c2a = tf.keras.layers.LeakyReLU(alpha=0.01)(b7_m_c2n)

    #Block 8 (Localization)
    b8_us = tf.keras.layers.Conv2DTranspose(32, (3,3), strides = (2,2), padding = 'same')(b7_m_c2a)
    b8_con = tf.keras.layers.concatenate([b8_us,b2_out])
    b8_m_c1 = tf.keras.layers.Conv2D(32, (3,3), padding = 'same')(b8_con)
    b8_m_c1n = tf.keras.layers.BatchNormalization()(b8_m_c1)
    b8_m_c1a = tf.keras.layers.LeakyReLU(alpha=0.01)(b8_m_c1n)
    b8_m_c2 = tf.keras.layers.Conv2D(32, (1,1), padding = 'same')(b8_m_c1a)
    b8_m_c2n = tf.keras.layers.BatchNormalization()(b8_m_c2)
    b8_m_c2a = tf.keras.layers.LeakyReLU(alpha=0.01)(b8_m_c2n)
    
    #Block 9 (Localization)
    b9_us = tf.keras.layers.Conv2DTranspose(16, (3,3), strides = (2,2), padding = 'same')(b8_m_c2a)
    b9_con = tf.keras.layers.concatenate([b9_us,b1_out])
    b9_m_c1 = tf.keras.layers.Conv2D(32, (3,3), padding = 'same')(b9_con)
    b9_m_c1n = tf.keras.layers.BatchNormalization()(b9_m_c1)
    b9_m_c1a = tf.keras.layers.LeakyReLU(alpha=0.01)(b9_m_c1n)

    output_layer = tf.keras.layers.Conv2D(2, (1,1), activation = 'softmax')(b9_m_c1a)
    return (input_layer, output_layer)