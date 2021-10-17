import tensorflow as tf
from model import VQ_VAE

# Fix memory growth issue encountered when using tensorflow
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

images = tf.random.uniform([10, 176, 176, 1])
vq_vae = VQ_VAE(176, 176, 1, n_encoded_features=96, embedding_dim=64, n_embeddings=128, train_variance=0.5)
# Print out the architecture of the encoder
vq_vae.encoder.summary()
# Print out the architecture of the decoder
vq_vae.decoder.summary()
