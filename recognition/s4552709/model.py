    # f  class = 4
def adv_model(f): 
    input_layer = layers.Input(shape= (h,w,1))
    # padding='same' ---> output_shape = input_shape / strides 
    # ensure the height and width of the output feature maps matches the inputs.
   

    # black convolution
    conv1 = layers.Conv2D(16, (3, 3), padding='same')(input_layer)
    conv1 = tf.keras.layers.LeakyReLU(0.01)(conv1)  
    # grey context model
    conv2 = layers.Conv2D(16, (3, 3),  padding='same')(conv1)
    conv2 = tf.keras.layers.LeakyReLU(0.01)(conv2)
    conv2 = layers.Dropout(0.3)(conv2)
    conv2 = layers.Conv2D(16, (3, 3),  padding='same')(conv2)
    conv2 = tf.keras.layers.LeakyReLU(0.01)(conv2)
 
    #element wise sum
    convsum1 = conv1 + conv2
    
   # gold 3*3 stride 2 convolution
    conv3 = layers.Conv2D(32, (3, 3), strides=2, padding = 'same')(convsum1)
    # grey context model
    conv4 = layers.Conv2D(32, (3, 3),  padding='same')(conv3)
    conv4 = tf.keras.layers.LeakyReLU(0.01)(conv4)
    conv4 = layers.Dropout(0.3)(conv4)
    conv4 = layers.Conv2D(32, (3, 3),  padding='same')(conv4)
    conv4 = tf.keras.layers.LeakyReLU(0.01)(conv4)
    
    convsum2 = conv3 + conv4
    
    # gold 3*3 stride 2 convolution
    conv5 = layers.Conv2D(64, (3, 3), strides= 2, padding = 'same')(convsum2)
    # grey context model
    conv6 = layers.Conv2D(64, (3, 3),  padding='same')(conv5)
    conv6 = tf.keras.layers.LeakyReLU(0.01)(conv6)
    conv6 = layers.Dropout(0.3)(conv6)
    conv6 = layers.Conv2D(64, (3, 3),  padding='same')(conv6)
    conv6 = tf.keras.layers.LeakyReLU(0.01)(conv6)
    
    convsum3 = conv5 + conv6
    
    
    # gold 3*3 stride 2 convolution
    conv7 = layers.Conv2D(128, (3, 3), strides=2, padding = 'same')(convsum3)
    # grey context model
    conv8 = layers.Conv2D(128, (3, 3),  padding='same')(conv7)
    conv8 = tf.keras.layers.LeakyReLU(0.01)(conv8)
    conv8 = layers.Dropout(0.3)(conv8)
    conv8 = layers.Conv2D(128, (3, 3),  padding='same')(conv8)
    conv8 = tf.keras.layers.LeakyReLU(0.01)(conv8)
    
    convsum4 = conv7 + conv8
    
    
    # gold 3*3 stride 2 convolution
    conv9 = layers.Conv2D(256, (3, 3), strides=2, padding = 'same')(convsum4)
    # grey context model
    conv10 = layers.Conv2D(256, (3, 3),  padding='same')(conv9)
    conv10 = tf.keras.layers.LeakyReLU(0.01)(conv10)
    conv10 = layers.Dropout(0.3)(conv10)
    conv10 = layers.Conv2D(256, (3, 3),  padding='same')(conv10)
    conv10 = tf.keras.layers.LeakyReLU(0.01)(conv10)
    
    convsum5 = conv9 + conv10
    
    # blue  upsampling module
    up11 = layers.UpSampling2D(size=(2, 2))(convsum5)
    # concatenation
    merge1 = layers.concatenate([convsum4, up11])
    
    # orange locaization 
    conv12 = layers.Conv2D(128, (3, 3),  padding='same')(merge1)
    conv12 = tf.keras.layers.LeakyReLU(0.01)(conv12)
    conv12 = layers.Conv2D(128, (1, 1),  padding='same')(conv12)
    conv12 = tf.keras.layers.LeakyReLU(0.01)(conv12)
    # blue  upsampling module
    up13 = layers.UpSampling2D(size=(2, 2))(conv12)
    # concatenation
    merge2 = layers.concatenate([convsum3, up13])
    
    # orange locaization 
    conv14 = layers.Conv2D(64, (3, 3),  padding='same')(merge2)
    conv14 = tf.keras.layers.LeakyReLU(0.01)(conv14)
    conv14 = layers.Conv2D(64, (1, 1),  padding='same')(conv14)
    conv14 = tf.keras.layers.LeakyReLU(0.01)(conv14)
    # segmentation layer 
    seg1 = layers.Conv2D(4, (1, 1),  padding='same')(conv14)
    seg1 = tf.keras.layers.LeakyReLU(0.01)(seg1)
    upscale1 = layers.UpSampling2D(size=(2, 2))(seg1)
                                   
    # blue  upsampling module
    up15 = layers.UpSampling2D(size=(2, 2))(conv14)
    merge3 = layers.concatenate([convsum2, up15])
    
    # orange locaization 
    conv16 = layers.Conv2D(32, (3, 3),  padding='same')(merge3)
    conv16 = tf.keras.layers.LeakyReLU(0.01)(conv16)
    conv16 = layers.Conv2D(32, (1, 1),  padding='same')(conv16)
    conv16 = tf.keras.layers.LeakyReLU(0.01)(conv16)
    # segmentation layer 
    seg2 = layers.Conv2D(4, (1, 1),  padding='same')(conv16)
    seg2 = tf.keras.layers.LeakyReLU(0.01)(seg2)
    convsum6 = upscale1 + seg2
    upscale2 = layers.UpSampling2D(size=(2, 2))(convsum6)
    # blue  upsampling module
    up17 = layers.UpSampling2D(size=(2, 2))(conv16)
    
    # concatenation
    merge4 = layers.concatenate([convsum1, up17])
    
    # black
    conv18 = layers.Conv2D(32, (3, 3),  padding='same')(merge4)
    conv18 = tf.keras.layers.LeakyReLU(0.01)(conv18) 
    # segmentation layer 
    seg3 = layers.Conv2D(4, (1, 1),  padding='same')(conv18)
    seg3 = tf.keras.layers.LeakyReLU(0.01)(seg3)
    convsum6 = upscale2 + seg3

    output_layer = layers.Conv2D(f, (1,1), padding="same", activation="softmax")(convsum6)
    
    
    model = tf.keras.Model(inputs= input_layer, outputs = output_layer)
    #Model.summary()

    # config the model with losses and metrics
    # use a dice loss function to cope with class imbalances 
    model.compile(optimizer='adam',
                  loss= 'categorical_crossentropy')

    return model