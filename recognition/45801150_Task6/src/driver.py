import numpy as np
import VQVAE
import load_oasis_data
import PixelCNN

latent_dimensions = 16
num_embeddings = 128
print(f"latent_dimensions: {latent_dimensions}, num_embeddings={num_embeddings}")

x_train, x_test, x_val = load_oasis_data.get_data(10)

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
x_val = np.expand_dims(x_val, -1)

x_train_normalised = (x_train / 255.0)
x_test_normalised = (x_test / 255.0)
x_val_normalised = (x_val / 255.0)

variance = np.var(x_train / 255.0)

vqvae = VQVAE.VQVae(variance, latent_dimensions, num_embeddings)
VQVAE.train_vqvae(vqvae, x_train_normalised, x_val_normalised, 2)
VQVAE.compare_reconstructions(vqvae, x_test_normalised, 10)

encoder = vqvae.get_layer("encoder")
encoder_output_shape = encoder.predict(x_test[0:1]).shape
pixel_cnn = PixelCNN.create_pixel_cnn(encoder_output_shape, num_embeddings)

PixelCNN.train_pixel_cnn(pixel_cnn, vqvae, x_train_normalised, 2)

PixelCNN.generate_image(vqvae, pixel_cnn, pixel_cnn.input_shape, encoder_output_shape)

