import numpy as np
import VQVAE
import load_oasis_data
import PixelCNN
import visualiser

## Hyper parameters
latent_dimensions = 16
num_embeddings = 128


def main():
    # Get and normalise data
    x_train, x_test, x_val = load_oasis_data.get_data()
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    x_val = np.expand_dims(x_val, -1)
    x_train_normalised = (x_train / 255.0)
    x_test_normalised = (x_test / 255.0)
    x_val_normalised = (x_val / 255.0)

    variance = np.var(x_train / 255.0)

    # Create and train VQ-VAE
    vqvae = VQVAE.VQVae(variance, latent_dimensions, num_embeddings)
    VQVAE.train_vqvae(vqvae, x_train_normalised, x_val_normalised, 2)

    # Test VQ-VAE performance on test set
    test_images, reconstructed = visualiser.compare_reconstructions(vqvae, x_test_normalised, 10)
    visualiser.show_reconstructions(10, test_images, reconstructed)

    # Create and train PixelCNN
    encoder = vqvae.get_layer("encoder")
    encoder_output_shape = encoder.predict(x_test[0:1]).shape
    pixel_cnn = PixelCNN.create_pixel_cnn(encoder_output_shape, num_embeddings)
    PixelCNN.train_pixel_cnn(pixel_cnn, vqvae, x_train_normalised, 2)

    # Generate images, testing PixelCNN performance
    codes, generated = PixelCNN.generate_images(vqvae, pixel_cnn, 10, encoder_output_shape)
    visualiser.show_generated_images(10, codes, generated)

if __name__ == "__main__":
    main()
