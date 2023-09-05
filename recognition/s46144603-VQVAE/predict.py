from train import *

# Comparison of original image with reconstructed image
def reconstruct_image(original, reconstructed):
    plt.subplot(1, 2, 1)
    plt.imshow(original.squeeze())
    plt.title("Original")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed.squeeze())
    plt.title("Reconstructed")
    plt.axis("off")
    plt.show()

# Find random image and create model prediction on test set
vqvae_model = train_vqvae.vqvae
index = np.random.choice(len(X_test), 3)
test_images = X_test[index]
reconstructions_test = vqvae_model.predict(test_images)

# Plots of random original vs reconstructed brain scans 
# Calculate SSIM for this image
for test_image, reconstructed_image in zip(test_images, reconstructions_test):
    reconstruct_image(test_image, reconstructed_image)
    print(tf.image.ssim(test_image, reconstructed_image, max_val=1))