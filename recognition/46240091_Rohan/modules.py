import numpy as np
from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow as tf
from tensorflow import keras
import keras.models
import keras.preprocessing.image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import tensorflow_probability as tfp
from keras import layers


def encoder(latent_dim=16):
  """
  Encoder Network for VQVAE
  """
  model = keras.Sequential(name="encoder");
  model.add(keras.Input(shape=(64, 64, 1)))
  model.add(layers.Conv2D(16, 3, activation="relu", strides=2, padding="same"))
  model.add(layers.Conv2D(32, 3, activation="relu", strides=2, padding="same"))
  model.add(layers.Conv2D(64, 3, activation="relu", strides=2, padding="same"))
  model.add(layers.Conv2D(latent_dim, 1, padding="same"))
  return model


def decoder(latent_dim=16):
  """
  Decoder Network for VQVAE
  """
  model = keras.Sequential(name="decoder");
  model.add(keras.Input(shape=encoder().output.shape[1:]))
  model.add(layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same"))
  model.add(layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same"))
  model.add(layers.Conv2DTranspose(16, 3, activation="relu", strides=2, padding="same"))
  model.add(layers.Conv2DTranspose(1, 3, padding="same"))
  return model

"""
VQVAE model based on the paper https://arxiv.org/abs/1711.00937
VectorQuantizer class from Keras tutorial on VQVAE https://keras.io/examples/generative/vq_vae/#vectorquantizer-layer
"""

class VectorQuantizer(layers.Layer):
  """
  Vector Quantizer custom layer according to https://keras.io/examples/generative/vq_vae/#vectorquantizer-layer
  """

  def __init__(self, num_embeddings, embedding_dim, **kwargs):
      super().__init__(**kwargs)
      self.embedding_dim = embedding_dim
      self.num_embeddings = num_embeddings
      #Beta = 1 found to give very good results in less number of epochs, should be kept between 0.25 and 2
      self.beta = 1
      
      #Initialising the embeddings randomly, which will be quantized and then L2 distance calculated between output from encoder
      #and entries from this.
      initial = tf.random_uniform_initializer()
      self.embeddings = tf.Variable(initial_value=initial(shape=(self.embedding_dim, self.num_embeddings), dtype="float32"),
          trainable=True,
          name="embeddings_vqvae",
      )

  def get_code_indices(self, flattened_inputs):
    """
    Calculating the L2 normalized distance between flattened input and codebook entries
    and then returning the indices with min distance
    """
    similarity = tf.matmul(flattened_inputs, self.embeddings)
    distances = (
        tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
        + tf.reduce_sum(self.embeddings ** 2, axis=0)
        - 2 * similarity
    )

    encoding_indices = tf.argmin(distances, axis=1)
    return encoding_indices

  def call(self, x):
    #Input shape calculated and inputs reshaped (embedding dimensions not flattened as it is size of latent embeddings)
    input_shape = tf.shape(x)
    flattened = tf.reshape(x, [-1, self.embedding_dim])

    #Applying one-hot encoding to code with min distance from flattened inputs
    encoding_indices = self.get_code_indices(flattened)
    encodings = tf.one_hot(encoding_indices, self.num_embeddings)
    quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
    quantized = tf.reshape(quantized, input_shape)

    #Calculating losses
    commitment_loss = self.beta * tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2) 
    codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
    self.add_loss(commitment_loss + codebook_loss)

    #Straight through estimator between decoder and encoder.
    quantized = x + tf.stop_gradient(quantized - x)
    return quantized


def vqvae_model(latent_dim=16, num_embeddings=64):
  """
  Vqvae model that combines the encoder, vq-layer, and decoder 
  """
  #Getting the 3 models
  encoder_model = encoder(latent_dim)
  decoder_model = decoder(latent_dim)
  vq_layer = VectorQuantizer(num_embeddings, latent_dim, name="vector_quantizer")
  #Giving the inputs to encoder and using it's output as input to vqlayer, vqlayers output used as input to decoder.
  encoder_inputs = keras.Input(shape=(64, 64, 1))
  vq_layer_inputs = encoder_model(encoder_inputs)
  decoder_inputs = vq_layer(vq_layer_inputs)
  vqvae_output = decoder_model(decoder_inputs)
  return keras.Model(encoder_inputs, vqvae_output, name="vq_vae")   


class VQVAETrainer(keras.models.Sequential):
  """
  VQVAE trainer class from Keras tutorial: 
  https://keras.io/examples/generative/vq_vae/#wrapping-up-the-training-loop-inside-vqvaetrainer
  """
  def __init__(self, train_variance, latent_dim=16, num_embeddings=128, **kwargs):
    super(VQVAETrainer, self).__init__(**kwargs)
    self.train_variance = train_variance
    self.latent_dim = latent_dim
    self.num_embeddings = num_embeddings
    self.vqvae1 = vqvae_model(self.latent_dim, self.num_embeddings)
    self.total = keras.metrics.Mean()
    self.reconstruction = keras.metrics.Mean()
    self.vq = keras.metrics.Mean()

  @property
  def metrics(self):
    """
    The 3 loss metrics, reconstruction loss represents the information lost between the
    original image and image reconstructed from VQVAE and total loss is the sum 
    of this loss and sum of all losses from vqvae_model
    """
    return [self.total, self.reconstruction, self.vq]

  def train_step(self, x):
    with tf.GradientTape() as tape:
      #Reconstructed images which are output from VQVAE.
      reconstructions = self.vqvae1(x)

      # Calculating the losses.
      reconstruction_loss = (tf.reduce_mean((x - reconstructions) ** 2) / self.train_variance)
      total_loss = reconstruction_loss + sum(self.vqvae1.losses)

    # Backpropagation.
    gradients = tape.gradient(total_loss, self.vqvae1.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.vqvae1.trainable_variables))

    #Updating the losses
    self.total.update_state(total_loss)
    self.reconstruction.update_state(reconstruction_loss)
    self.vq.update_state(sum(self.vqvae1.losses))

    return {
    "total_loss": self.total.result(),
    "reconstruction_loss": self.reconstruction.result(),
    "vqvae_loss": self.vq.result(),
    }


#######PIXELCNN#######

class MaskedConv2D(keras.layers.Layer):
  """
  This class creates conv layer with mask type A and B for my
  autoregressive model: pixelcnn
  """
  def __init__(self,
              mask_type,
               **kwargs):
    super(MaskedConv2D, self).__init__()
    assert mask_type in {'A', 'B'}
    self.mask_type = mask_type
    self.conv = layers.Conv2D(**kwargs)

  def build(self, input_shape):
    #Initializing the kernel and using it to create masks
    self.conv.build(input_shape)
    kernel_shape = tf.shape(self.conv.kernel)
    self.kernel_size = kernel_shape[0]

    center = self.kernel_size // 2
    mask = np.ones(kernel_shape, dtype=np.float32)
    #Condition about mask_type = 'B' valid when its not the first layer of Pixelcnn
    mask[center, center + (self.mask_type == 'B'):, :, :] = 0.
    mask[center + 1:, :, :, :] = 0.
    self.mask = tf.constant(mask, dtype=tf.float32, name='mask')

  def call(self, input):
    #Applying the mask to the kernel
    self.conv.kernel.assign(self.conv.kernel * self.mask)
    return self.conv(input)



class ResidualBlock(keras.Model):
  """
  Layer blocks which have 3 convolutional layers with one residual connection. The 
  masked conv layer has a standard conv layer before and after.
  """

  def __init__(self, num_filters):
    super(ResidualBlock, self).__init__()

    self.conv2a = keras.layers.Conv2D(filters=num_filters, kernel_size=1, strides=1, activation="relu", padding='same')
    self.conv2b = MaskedConv2D(mask_type='B', filters=num_filters, kernel_size=3, strides=1, activation="relu", padding='same')
    self.conv2c = keras.layers.Conv2D(filters=2 * num_filters, kernel_size=1, strides=1, activation="relu", padding='same')

  def call(self, input_tensor):
    a = self.conv2a(input_tensor)
    a = self.conv2b(a)
    a = self.conv2c(a)
    return keras.layers.add([input_tensor, a])


def pcnn_model_maker(input_shape, existing_vqvae):
    """
    Making the model as mentioned in the original Pixel Recurrent Neural Networks paper
    The first layer is a masked convolution (type A) with 7x7 filters. Then, 15 residuals blocks were used.
    Then a chain of Relu X Conv 
    Then the output layer.

    """

    pixelcnn_inputs = keras.Input(shape=input_shape, dtype=tf.int32)
    ohe = tf.one_hot(pixelcnn_inputs, existing_vqvae.num_embeddings)
    #First layer (mask type A)
    l = MaskedConv2D(
        mask_type="A", filters=128, kernel_size=7, activation="relu", padding="same")(ohe)

    #15 residual blocks according to original pixelcnn paper (mask type B)
    for i in range(15):
        l = ResidualBlock(num_filters=64)(l)

    
    #Chain of Relu X Conv 
    l = keras.layers.Activation(activation='relu')(l)
    l = keras.layers.Conv2D(filters=128, kernel_size=1, strides=1)(l)
    l = keras.layers.Activation(activation='relu')(l)
    l = keras.layers.Conv2D(filters=128, kernel_size=1, strides=1)(l)
    l = keras.layers.Conv2D(filters=existing_vqvae.num_embeddings, kernel_size=1, strides=1)(l)

    return keras.Model(inputs=pixelcnn_inputs, outputs=l, name = 'pixelcnn')