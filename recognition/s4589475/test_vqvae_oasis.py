import tensorflow as tf


from model.py import *
from driver.py import *


if __name__ == '__main__':
    #Load the OASIS Dataset
    batch_size = 50
    training_ds, testing_ds_unbatched, testing_ds_batched = create_train_test_dataset('H:\keras_png_slices_data\keras_png_slices_train\*','H:\keras_png_slices_data\keras_png_slices_test\*', batch_size)

    # Create the model
    latent_dims = 128
    #VQ-VAE parameters based on the van den Oord et al. paper
    beta = 0.25 #commitment loss weighting
    K = 512 #Number of codebook vectors / embeddings number (K)

    #Create the encoder and decoder components of VQ-VAE Model
    encoder = encoder_network(latent_dims)
    decoder = decoder_network(latent_dims)
    quantizer_layer = Vector_Quantizer(latent_dims, K)
    optimizer = tf.keras.optimizers.Adam(1e-4)

    vq_vae_overall = create_overall_vqvae(encoder, quantizer_layer, decoder)

    # Train the model
    epochs = 30
    train(encoder, decoder, quantizer_layer, optimizer, vq_vae_overall, beta, epochs)

    #Display the mean ssim
    ssim_score = calculate_ssims(vq_vae_overall)
    print("Average SSIM on the test dataset:"+ ssim_score)

    # Show 10 example latent code images
    encoded_outputs = encoder.predict(testing_ds_unbatched)
    flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
    codebook_indices, vectors = quantizer_layer.quantize_vectors(flat_enc_outputs)
    codebook_indices2 = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])

    for i, image in enumerate(testing_ds_unbatched):
        plt.subplot(1, 2, 1)
        plt.imshow(image[0, :, :, 0], cmap='gray')
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(codebook_indices2[i])
        plt.title("Code")
        plt.axis("off")
        plt.show()
        
        if i > 10:
            break

    # Generate novel images using pixel CNN - based largely on the tutorial from https://keras.io/examples/generative/pixelcnn/
    # PixelCNN model parameters
    num_residual_blocks = 2
    num_pixelcnn_layers = 2

    # Create the pixel CNN model
    pixel_cnn = create_pixelCNN()

    # Print a summary of the model architecture
    pixel_cnn.summary()

    # Train the pixel_cnn for 500 epochs
    train_pixel_cnn(pixel_cnn, 500, training_ds, encoder, quantizer_layer)

    # Display 5 generated latent codes and corresponding image from the pixelCNN
    display_generated_images(5, pixel_cnn, quantizer_layer, decoder)
