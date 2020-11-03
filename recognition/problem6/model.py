from tensorflow.keras.layers import Input, Reshape, Dropout, Dense 
from tensorflow.keras.layers import Flatten, BatchNormalization
from tensorflow.keras.layers import Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam

tfk2 = tf.keras.layers



def build_discriminator(input_shape):
    
    print(input_shape)
    input_layer  = tf.keras.Input(input_shape)
    t = tfk2.Conv2D(32, 3, padding = "same", activation = tf.nn.leaky_relu)(input_layer)
    t = tfk2.Conv2D(32, 3, padding = "same", activation = tf.nn.leaky_relu)(t)
    t = tfk2.MaxPool2D()(t)

    t = tfk2.Conv2D(64, 3, padding = "same", activation = tf.nn.leaky_relu)(t)
    t = tfk2.Conv2D(64, 3, padding = "same", activation = tf.nn.leaky_relu)(t)
    t = tfk2.MaxPool2D()(t)

    t = tfk2.Conv2D(128, 3, padding = "same", activation = tf.nn.leaky_relu)(t)
    t = tfk2.Conv2D(128, 3, padding = "same", activation = tf.nn.leaky_relu)(t)
    t = tfk2.MaxPool2D()(t)

    t = tfk2.Conv2D(256, 3, padding = "same", activation = tf.nn.leaky_relu)(t)
    t = tfk2.Conv2D(256, 3, padding = "same", activation = tf.nn.leaky_relu)(t)
    t = tfk2.MaxPool2D()(t)

    t = tfk2.Flatten()(t)
    t = tfk2.Dense(256, activation = tf.nn.leaky_relu)(t)
    t = tfk2.Dense(256, activation = tf.nn.leaky_relu)(t)

    t = tfk2.Dense(1, activation='sigmoid')(t)

    model = tf.keras.Model(inputs = input_layer, outputs = t)
    model.summary()
    return model



def build_generator(input_shape):
  
    input_layer  = tf.keras.Input(input_shape)
    t = tfk2.Dense(16*16*256)(input_layer)
    t = tfk2.Reshape((16,16,256))(t)

    t = tfk2.UpSampling2D()(t)
    t = tfk2.Conv2D(256, 3, padding = "same", activation = tf.nn.leaky_relu)(t)

    t = tfk2.UpSampling2D()(t)
    t = tfk2.Conv2D(64, 3, padding = "same", activation = tf.nn.leaky_relu)(t)

    t = tfk2.UpSampling2D()(t)
    t = tfk2.Conv2D(3, 3, padding = "same", activation = "tanh")(t)

    t = tfk2.UpSampling2D()(t)
    t = tfk2.Conv2D(3, 3, padding = "same", activation = "tanh")(t)
    model = tf.keras.Model(inputs = input_layer, outputs = t)
    model.summary()
    return model
