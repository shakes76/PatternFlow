from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
from dataset import *
from modules import *
from predict import *
from train import *

VQVAE_DIR = "vqvae"
PIXELCNN_DIR = "pixelcnn"
LATENT_DIM = 16
NUM_EMBEDDINGS = 64
RESIDUAL_HIDDENS = 128
EPOCHS = 50
BATCH_SIZE = 64
DATA_DIR = 'data/keras_png_slices_data'
TRAIN_DATA = DATA_DIR + '/keras_png_slices_train'
TEST_DATA = DATA_DIR + '/keras_png_slices_test'
VALIDATE_DATA = DATA_DIR + '/keras_png_slices_validate'
#model = tf.keras.models.load_model(VQVAE_DIR, custom_objects={'VectorQuantizer': VectorQuantizer})
#pixelcnn = tf.keras.models.load_model(PIXELCNN_DIR, custom_objects={'PixelCNN': PixelCNN, 'ResidualBlock': ResidualBlock})

x_train, x_test, x_validate = get_data(TRAIN_DATA, TEST_DATA, VALIDATE_DATA)
model = train(x_train, x_test, x_validate, 
    epochs=EPOCHS, batch_size=BATCH_SIZE, out_dir=VQVAE_DIR,
    latent_dim=LATENT_DIM, 
    num_embeddings=NUM_EMBEDDINGS,
    residual_hiddens=RESIDUAL_HIDDENS
)

demo_vqvae(model,  x_test)

pixelcnn = pixelcnn_train(model, x_train, x_test, x_validate, 
    epochs=EPOCHS, batch_size=BATCH_SIZE, out_dir=PIXELCNN_DIR,
    num_embeddings=NUM_EMBEDDINGS
)
sample_images(model, pixelcnn)
