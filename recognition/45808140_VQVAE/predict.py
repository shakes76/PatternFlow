from dataset import *
from modules import *

def show_reconstructed(original, reconstructed, codebook_indices):
    plt.subplot(1, 3, 1)
    plt.imshow(original.numpy().squeeze())
    plt.title("Original")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(codebook_indices)
    plt.title("Code")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(reconstructed.squeeze())
    plt.title("Reconstructed")
    plt.axis("off")
    plt.show()

def VQVAE_result(vqvae, dataset):
    test_images = None
    for i in dataset.take(1):
        test_images = i

    encoder = vqvae.get_encoder()
    vq = vqvae.get_vq()
    recons = vq.predict(test_images)

    encoded_outputs = encoder.predict(test_images)
    flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
    codebook_ind = vq.get_code_indices(flat_enc_outputs)
    codebook_ind = codebook_ind.numpy().reshape(encoded_outputs.shape[:-1])

    for test_image, reconstructed_image, codebook in zip(test_images, recons, codebook_ind):
        show_reconstructed(test_image, reconstructed_image, codebook)


