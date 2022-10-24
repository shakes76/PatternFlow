from dataset import *

# Encoder
def encoder_net(latent_dim=32):
  input = tf.keras.layers.Input(input_shape, name="encoder_in")

  # Convolutional layers
  net = tf.keras.layers.Conv2D(depth, 3, padding='same', strides=2, activation='relu')(input)
  net = tf.keras.layers.Conv2D(depth*2, 3, padding='same', strides=2, activation='relu')(net)

  enc_out = tf.keras.layers.Conv2D(latent_dim, 1, padding='same')(net)
  
  return tf.keras.Model(inputs=input, outputs=enc_out, name='encoder')


