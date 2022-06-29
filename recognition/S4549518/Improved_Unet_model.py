#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# According to the essay, we need to wirte the function to build context modules, 
#localization modules and upsampling modules. Follow the essay, we use LeakyReLU as activation function.

leakyReLU =tf.keras.layers.LeakyReLU(alpha=1e-2)
        
def context_module(num_filters,inp):
    '''
    The context module:
    
     Arguments:
        num_filters{int} -- the number of filters
        inp {keras layer} -- input layer 
    
    Returns:
        [keras layer] -- [output layer]
    '''

    c1 = tfa.layers.InstanceNormalization()(inp)
    c1 = tf.keras.layers.Activation("relu")(c1)
    c2 = tf.keras.layers.Conv2D(num_filters, (3, 3), activation = leakyReLU, padding="same")(c1)
    c3 = tf.keras.layers.Dropout(0.3)(c2)
    c4 = tfa.layers.InstanceNormalization()(c3)
    c4 = tf.keras.layers.Activation("relu")(c4)
    c4 = tf.keras.layers.Conv2D(num_filters, (3, 3), activation = leakyReLU, padding="same")(c4)
    

    return c4       

def localization_modules(num_filters,inp):
    '''
    The context module:
    
     Arguments:
        num_filters{int} -- the number of filters
        inp {keras layer} -- input layer 
    
    Returns:
        [keras layer] -- [output layer]
    '''

    c1 = tf.keras.layers.Conv2D(num_filters, (3, 3), activation = leakyReLU, padding="same")(inp)
    c2 = tf.keras.layers.Conv2D(num_filters, (1, 1), activation = leakyReLU, padding="same")(c1)

    return c2
    
def upsampling_modules(num_filters,inp):
  '''
  The context module:

   Arguments:
      num_filters{int} -- the number of filters
      inp {keras layer} -- input layer 

  Returns:
      [keras layer] -- [output layer]
  '''

  c1 = tf.keras.layers.UpSampling2D()(inp)
  c2 = tf.keras.layers.Conv2D(num_filters,(3,3), activation = leakyReLU, padding="same")(c1)

  return c2

# Build model
def improved_unet(height, width, n_channels):
  '''
  The improved model:

   Arguments:
      height{int} -- the height of image
      width{int} -- the width of image
      n_channels{int} -- the chanels of image 

  Returns:
      Improved Unet model
  '''
    inputs = tf.keras.layers.Input((height, width, n_channels))
    
    # Constracting
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation=leakyReLU, padding='same') (inputs)
    c2 = context_module(16, c1)
    c3 = tf.keras.layers.Add()([c1, c2])

    c4 = tf.keras.layers.Conv2D(32, (3, 3),strides=(2, 2), activation=leakyReLU, padding='same') (c3)
    c5 = context_module(32, c4)
    c6 = tf.keras.layers.Add()([c4, c5])

    c7 = tf.keras.layers.Conv2D(64, (3, 3),strides=(2, 2), activation=leakyReLU, padding='same') (c6)
    c8 = context_module(64, c7)
    c9 = tf.keras.layers.Add()([c7, c8])

    c10 = tf.keras.layers.Conv2D(128, (3, 3),strides=(2, 2), activation=leakyReLU, padding='same') (c9)
    c11 = context_module(128, c10)
    c12 = tf.keras.layers.Add()([c10, c11])

    c13 = tf.keras.layers.Conv2D(256, (3, 3),strides=(2, 2), activation=leakyReLU, padding='same') (c12)
    c14 = context_module(256, c13)
    c15 = tf.keras.layers.Add()([c13, c14])
    
    # Expanding
    u16 = upsampling_modules(128,c15)
    
    con17 = tf.keras.layers.concatenate([u16, c12], axis=3)
    c18 = localization_modules(128,con17)
    u19 = upsampling_modules(64,c18)
    
    con20 = tf.keras.layers.concatenate([u19, c9], axis=3)
    c21 = localization_modules(64,con20)
    wise1 = tf.keras.layers.Conv2D(1, (1, 1), activation = leakyReLU, padding="same")(c21)
    up_wise1 = tf.keras.layers.UpSampling2D(interpolation = "bilinear")(wise1)
    u22 = upsampling_modules(32,c21)
    
    con23 = tf.keras.layers.concatenate([u22, c6], axis=3)
    c24 = localization_modules(32,con23)
    wise2 = tf.keras.layers.Conv2D(1, (1, 1), activation = leakyReLU, padding="same")(c24)
    wise2 = tf.keras.layers.Add()([up_wise1, wise2])
    up_wise2 = tf.keras.layers.UpSampling2D(interpolation = "bilinear")(wise2)
    u25 = upsampling_modules(16,c24)
    
    con26 = tf.keras.layers.concatenate([u25, c3], axis=3)
    c27 = tf.keras.layers.Conv2D(32, (3, 3), activation = leakyReLU, padding='same')(con26)
    c28 = tf.keras.layers.Conv2D(1, (1, 1), activation = leakyReLU, padding="same")(c27) 
    wise3 = tf.keras.layers.Add()([up_wise2, c28])
    
    outputs = tf.keras.activations.sigmoid(wise3)
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    
    return model    
   
if __name__ == "__main__":
    model = improved_unet()
    model.summary()

