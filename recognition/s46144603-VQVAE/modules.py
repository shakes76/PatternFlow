from dataset import *

# Encoder
def encoder_net(latent_dim=32):
  input = tf.keras.layers.Input(input_shape, name="encoder_in")

  # Convolutional layers
  net = tf.keras.layers.Conv2D(depth, 3, padding='same', strides=2, activation='relu')(input)
  net = tf.keras.layers.Conv2D(depth*2, 3, padding='same', strides=2, activation='relu')(net)

  enc_out = tf.keras.layers.Conv2D(latent_dim, 1, padding='same')(net)
  
  return tf.keras.Model(inputs=input, outputs=enc_out, name='encoder')


# Build decoder
def decoder_net(latent_dim=32):
  decoder_in = tf.keras.Input(shape=encoder_net(latent_dim).output.shape[1:])

  net = tf.keras.layers.Conv2DTranspose(depth*2, 3, padding='same', strides=2, activation='relu')(decoder_in)
  net = tf.keras.layers.Conv2DTranspose(depth, 3, padding='same', strides=2, activation='relu')(net)
  network = tf.keras.layers.Conv2DTranspose(1, 3, padding='same')(net)

  return tf.keras.Model(inputs=decoder_in, outputs=network, name='decoder')