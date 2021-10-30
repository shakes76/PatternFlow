import tensorflow as tf
from model import VQ_VAE
import matplotlib.pyplot as plt
from paths import TEST_DATA_PATH, TRAINED_VQ_PATH
# Fix memory growth issue encountered when using tensorflow
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Define parameters for the data loader
h = 256
w = 256

test_loader = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DATA_PATH,
    label_mode=None,
    seed=0,
    color_mode='grayscale',
    image_size=(h, w),
    shuffle=True
)

# Normalise test data b4 feed it into the model
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1./255, offset=-0.5)
normalized_test_loader = test_loader.map(lambda x: normalization_layer(x))

# Load the model
vq_vae_wrapper = VQ_VAE(img_h=h, img_w=w, img_c=1, train_variance=0.0347, embedding_dim=24, n_embeddings=256, recon_loss_type='MSE', commitment_factor=2)
vq_vae_wrapper.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3))
#vq_vae_wrapper.vq_vae = tf.keras.models.load_model('vq_vae')
vq_vae_wrapper.load_weights(TRAINED_VQ_PATH)

# Compute SSIM of the trained model based on the test data
vq_vae_wrapper.evaluate(normalized_test_loader)

# Visualize some test images being reconstructed via VQ VAE
imgs = next(iter(normalized_test_loader))
vq_vae = vq_vae_wrapper.vq_vae
reconstructed_imgs = vq_vae.predict(imgs)
n_images = 4
# Plot the test images and their reconstructed versions
fig, axs = plt.subplots(n_images, 2, figsize=(50, 50))
for i in range(n_images):
    axs[i, 0].imshow(imgs[i], cmap=plt.cm.gray)
    axs[i, 0].set_title('Original version of image {}'.format(i+1))
    axs[i, 1].imshow(reconstructed_imgs[i], cmap=plt.cm.gray)
    axs[i, 1].set_title('Reconstructed version of image {}'.format(i+1))
    axs[i, 0].axis('off')
    axs[i, 1].axis('off')
plt.show()
