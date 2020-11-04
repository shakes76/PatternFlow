import tensorflow as tf

#Improved U-net model
# =============================================================================
# Layer naming:
# b# = block #
# ds = downsample
# res = residual
# m = module (context or localization)
# c# = convolution # in module
# c#n = normalization of conv #
# c#a = activation function of conv #
# do = dropout
# out = output (with res)
# us = upsample
# con = concatenation
# seg = segmentation
# =============================================================================
def ImprovedUnet(h, w, n_channels):
    F = [8,16,32,64,128]
    wd = 0.0005
    lrelu_alp = 0.01
    input_layer = tf.keras.layers.Input(shape = (h,w,n_channels))
        
    #Block 1 (Context)
    b1_c0 = tf.keras.layers.Conv2D(F[0], (3,3), padding = 'same', 
                                   kernel_regularizer = tf.keras.regularizers.l2(wd))(input_layer)
    b1_res = b1_c0
    b1_m_c1n = tf.keras.layers.BatchNormalization()(b1_c0)
    b1_m_c1a = tf.keras.layers.LeakyReLU(alpha=lrelu_alp)(b1_m_c1n)
    b1_m_c1 = tf.keras.layers.Conv2D(F[0], (3,3), padding = 'same', 
                                     kernel_regularizer = tf.keras.regularizers.l2(wd))(b1_m_c1a)
    b1_m_do = tf.keras.layers.Dropout(rate=0.3)(b1_m_c1)
    b1_m_c2n = tf.keras.layers.BatchNormalization()(b1_m_do)
    b1_m_c2a = tf.keras.layers.LeakyReLU(alpha=lrelu_alp)(b1_m_c2n)
    b1_m_c2 = tf.keras.layers.Conv2D(F[0], (3,3), padding = 'same', 
                                     kernel_regularizer = tf.keras.regularizers.l2(wd))(b1_m_c2a)
    b1_out = b1_m_c2 + b1_res
    
    #Block 2 (Context)
    b2_ds = tf.keras.layers.Conv2D(F[1], (3,3), strides = (2,2), padding = 'same')(b1_out)
    b2_res = b2_ds
    b2_m_c1n = tf.keras.layers.BatchNormalization()(b2_ds)
    b2_m_c1a = tf.keras.layers.LeakyReLU(alpha=lrelu_alp)(b2_m_c1n)
    b2_m_c1 = tf.keras.layers.Conv2D(F[1], (3,3), padding = 'same', 
                                     kernel_regularizer = tf.keras.regularizers.l2(wd))(b2_m_c1a)
    b2_m_do = tf.keras.layers.Dropout(rate=0.3)(b2_m_c1)
    b2_m_c2n = tf.keras.layers.BatchNormalization()(b2_m_do)
    b2_m_c2a = tf.keras.layers.LeakyReLU(alpha=lrelu_alp)(b2_m_c2n)
    b2_m_c2 = tf.keras.layers.Conv2D(F[1], (3,3), padding = 'same', 
                                     kernel_regularizer = tf.keras.regularizers.l2(wd))(b2_m_c2a)
    b2_out = b2_m_c2 + b2_res
    
    #Block 3 (Context)
    b3_ds = tf.keras.layers.Conv2D(F[2], (3,3), strides = (2,2), padding = 'same')(b2_out)
    b3_res = b3_ds
    b3_m_c1n = tf.keras.layers.BatchNormalization()(b3_ds)
    b3_m_c1a = tf.keras.layers.LeakyReLU(alpha=lrelu_alp)(b3_m_c1n)
    b3_m_c1 = tf.keras.layers.Conv2D(F[2], (3,3), padding = 'same', 
                                     kernel_regularizer = tf.keras.regularizers.l2(wd))(b3_m_c1a)
    b3_m_do = tf.keras.layers.Dropout(rate=0.3)(b3_m_c1)
    b3_m_c2n = tf.keras.layers.BatchNormalization()(b3_m_do)
    b3_m_c2a = tf.keras.layers.LeakyReLU(alpha=lrelu_alp)(b3_m_c2n)
    b3_m_c2 = tf.keras.layers.Conv2D(F[2], (3,3), padding = 'same', 
                                     kernel_regularizer = tf.keras.regularizers.l2(wd))(b3_m_c2a)
    b3_out = b3_m_c2 + b3_res
    
    #Block 4 (Context)
    b4_ds = tf.keras.layers.Conv2D(F[3], (3,3), strides = (2,2), padding = 'same')(b3_out)
    b4_res = b4_ds
    b4_m_c1n = tf.keras.layers.BatchNormalization()(b4_ds)
    b4_m_c1a = tf.keras.layers.LeakyReLU(alpha=lrelu_alp)(b4_m_c1n)
    b4_m_c1 = tf.keras.layers.Conv2D(F[3], (3,3), padding = 'same', 
                                     kernel_regularizer = tf.keras.regularizers.l2(wd))(b4_m_c1a)
    b4_m_do = tf.keras.layers.Dropout(rate=0.3)(b4_m_c1)
    b4_m_c2n = tf.keras.layers.BatchNormalization()(b4_m_do)
    b4_m_c2a = tf.keras.layers.LeakyReLU(alpha=lrelu_alp)(b4_m_c2n)
    b4_m_c2 = tf.keras.layers.Conv2D(F[3], (3,3), padding = 'same', 
                                     kernel_regularizer = tf.keras.regularizers.l2(wd))(b4_m_c2a)
    b4_out = b4_m_c2 + b4_res
    
    #Block 5 (Context)
    b5_ds = tf.keras.layers.Conv2D(F[4], (3,3), strides = (2,2), padding = 'same')(b4_out)
    b5_res = b5_ds
    b5_m_c1n = tf.keras.layers.BatchNormalization()(b5_ds)
    b5_m_c1a = tf.keras.layers.LeakyReLU(alpha=lrelu_alp)(b5_m_c1n)
    b5_m_c1 = tf.keras.layers.Conv2D(F[4], (3,3), padding = 'same', 
                                     kernel_regularizer = tf.keras.regularizers.l2(wd))(b5_m_c1a)
    b5_m_do = tf.keras.layers.Dropout(rate=0.3)(b5_m_c1)
    b5_m_c2n = tf.keras.layers.BatchNormalization()(b5_m_do)
    b5_m_c2a = tf.keras.layers.LeakyReLU(alpha=lrelu_alp)(b5_m_c2n)
    b5_m_c2 = tf.keras.layers.Conv2D(F[4], (3,3), padding = 'same', 
                                     kernel_regularizer = tf.keras.regularizers.l2(wd))(b5_m_c2a)
    b5_out = b5_m_c2 + b5_res
    
    #Block 6 (Localization)
    b6_us = tf.keras.layers.Conv2DTranspose(F[3], (3,3), strides = (2,2), padding = 'same')(b5_out)
    b6_us = tf.keras.layers.BatchNormalization()(b6_us)
    b6_us = tf.keras.layers.LeakyReLU(alpha=lrelu_alp)(b6_us)
    b6_con = tf.keras.layers.concatenate([b6_us,b4_out])
    b6_m_c1 = tf.keras.layers.Conv2D(F[4], (3,3), padding = 'same', 
                                     kernel_regularizer = tf.keras.regularizers.l2(wd))(b6_con)
    b6_m_c1n = tf.keras.layers.BatchNormalization()(b6_m_c1)
    b6_m_c1a = tf.keras.layers.LeakyReLU(alpha=lrelu_alp)(b6_m_c1n)
    b6_m_c2 = tf.keras.layers.Conv2D(F[3], (1,1))(b6_m_c1a)
    b6_m_c2n = tf.keras.layers.BatchNormalization()(b6_m_c2)
    b6_m_c2a = tf.keras.layers.LeakyReLU(alpha=lrelu_alp)(b6_m_c2n)
    
    #Block 7 (Localization)
    b7_us = tf.keras.layers.Conv2DTranspose(F[2], (3,3), strides = (2,2), padding = 'same')(b6_m_c2a)
    b7_us = tf.keras.layers.BatchNormalization()(b7_us)
    b7_us = tf.keras.layers.LeakyReLU(alpha=lrelu_alp)(b7_us)
    b7_con = tf.keras.layers.concatenate([b7_us,b3_out])
    b7_m_c1 = tf.keras.layers.Conv2D(F[3], (3,3), padding = 'same', 
                                     kernel_regularizer = tf.keras.regularizers.l2(wd))(b7_con)
    b7_m_c1n = tf.keras.layers.BatchNormalization()(b7_m_c1)
    b7_m_c1a = tf.keras.layers.LeakyReLU(alpha=lrelu_alp)(b7_m_c1n)
    b7_seg = b7_m_c1a
    b7_m_c2 = tf.keras.layers.Conv2D(F[2], (1,1))(b7_m_c1a)
    b7_m_c2n = tf.keras.layers.BatchNormalization()(b7_m_c2)
    b7_m_c2a = tf.keras.layers.LeakyReLU(alpha=lrelu_alp)(b7_m_c2n)

    #Block 8 (Localization)
    b8_us = tf.keras.layers.Conv2DTranspose(F[1], (3,3), strides = (2,2), padding = 'same')(b7_m_c2a)
    b8_us = tf.keras.layers.BatchNormalization()(b8_us)
    b8_us = tf.keras.layers.LeakyReLU(alpha=lrelu_alp)(b8_us)
    b8_con = tf.keras.layers.concatenate([b8_us,b2_out])
    b8_m_c1 = tf.keras.layers.Conv2D(F[2], (3,3), padding = 'same', 
                                     kernel_regularizer = tf.keras.regularizers.l2(wd))(b8_con)
    b8_m_c1n = tf.keras.layers.BatchNormalization()(b8_m_c1)
    b8_m_c1a = tf.keras.layers.LeakyReLU(alpha=lrelu_alp)(b8_m_c1n)
    b8_seg = b8_m_c1a
    b8_m_c2 = tf.keras.layers.Conv2D(F[1], (1,1))(b8_m_c1a)
    b8_m_c2n = tf.keras.layers.BatchNormalization()(b8_m_c2)
    b8_m_c2a = tf.keras.layers.LeakyReLU(alpha=lrelu_alp)(b8_m_c2n)
    
    #Block 9 (Localization)
    b9_us = tf.keras.layers.Conv2DTranspose(F[0], (3,3), strides = (2,2), padding = 'same')(b8_m_c2a)
    b9_us = tf.keras.layers.BatchNormalization()(b9_us)
    b9_us = tf.keras.layers.LeakyReLU(alpha=lrelu_alp)(b9_us)
    b9_con = tf.keras.layers.concatenate([b9_us,b1_out])
    b9_m_c1 = tf.keras.layers.Conv2D(F[1], (3,3), padding = 'same', 
                                     kernel_regularizer = tf.keras.regularizers.l2(wd))(b9_con)
    b9_m_c1n = tf.keras.layers.BatchNormalization()(b9_m_c1)
    b9_m_c1a = tf.keras.layers.LeakyReLU(alpha=lrelu_alp)(b9_m_c1n)
    b9_seg = tf.keras.layers.Conv2D(1, (1,1))(b9_m_c1a)
    
    #Block 10 (Segmentations)
    b7_seg = tf.keras.layers.Conv2D(1, (1,1))(b7_seg)
    b7_seg = tf.keras.layers.Conv2DTranspose(1, (3,3), strides = (2,2), padding = 'same')(b7_seg)
    b8_seg = tf.keras.layers.Conv2D(1, (1,1))(b8_seg)
    b78_seg = b7_seg + b8_seg
    b78_seg = tf.keras.layers.Conv2DTranspose(1, (3,3), strides = (2,2), padding = 'same')(b78_seg)
    b789_seg = b78_seg + b9_seg
    
    output_layer = tf.keras.layers.Conv2D(1, (1,1), activation = 'sigmoid')(b789_seg)
    return tf.keras.Model(inputs = input_layer, outputs = output_layer)