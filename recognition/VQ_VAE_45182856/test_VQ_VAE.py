import tensorflow as tf
from model import VQ_VAE

# Fix memory growth issue encountered when using tensorflow
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Define directory
img_folder = 'D:/UQ/Fourth Sem/Pattern-Analysis/Prac2/keras_png_slices_data/'

# Define parameters for the data loader
batch_size = 128
h = 176
w = 176
train_loader = tf.keras.preprocessing.image_dataset_from_directory(
    img_folder,
    label_mode=None,
    seed=123,
    color_mode='grayscale',
    image_size=(h, w),
    shuffle=True,
    batch_size=batch_size
)

# Normalise training data b4 feed it into the model
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1./255, offset=-0.5)
normalized_train_loader = train_loader.map(lambda x: normalization_layer(x))

vq_vae = VQ_VAE(h, w, 1, n_encoded_features=96, embedding_dim=64, n_embeddings=128)
# Print out the architecture of the encoder
vq_vae.encoder.summary()
# Print out the architecture of the decoder
vq_vae.decoder.summary()

optimizer = tf.keras.optimizers.Adam(lr=1e-3)
vq_vae.compile(optimizer=optimizer)
vq_vae.fit(normalized_train_loader, epochs=2)