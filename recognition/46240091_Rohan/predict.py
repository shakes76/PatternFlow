import os
from dataset import *
from modules import *
from train import *

OASIS_TRAIN_DIR = os.path.join(os.curdir, "keras_png_slices_data", "keras_png_slices_train")
OASIS_TEST_DIR = os.path.join(os.curdir, "keras_png_slices_data", "keras_png_slices_test")


NUM_LATENT_DIMS = 16
NUM_EMBEDDINGS = 128

#Loading and preprocesing the data 
xnotscaled = data_loader(OASIS_TRAIN_DIR, scale_flag=False)
x = data_loader(OASIS_TRAIN_DIR,scale_flag=True)
xt = data_loader(OASIS_TEST_DIR, scale_flag=True)

#Variance in training images
data_variance = np.var(xnotscaled / 255.0)
#Training data dims
print("This is training data shape: ", x.shape)


#VQVAE training
vqvae, history = vqvae_training(training_data = x, data_variance = data_variance)
#Plotting the training loss
VQVAE_training_plot(history)


#Plotting the actual images with reconstructed ones along with ssim
print("Showing actual images with reconstructed images")
trained_vqvae_model = vqvae.vqvae1
# 10 random test images
idx = np.random.choice(len(xt), 10)
test_images = xt[idx]
# Perform predictions on test images
reconstructions_test = trained_vqvae_model.predict(test_images)
total = 0
for test_image, reconstructed_image in zip(test_images, reconstructions_test):
  plt.subplot(1, 2, 1)
  plt.imshow(test_image.squeeze(), cmap=plt.cm.gray)
  plt.title("Test Image")
  plt.axis("off")

  plt.subplot(1, 2, 2)
  plt.imshow(reconstructed_image.squeeze(), cmap=plt.cm.gray)
  plt.title("Reconstructed Test Image")
  plt.axis("off")
  plt.show()
  total += tf.image.ssim(test_image, reconstructed_image, max_val=1)
print("Average SSIM = ", total/10)  








