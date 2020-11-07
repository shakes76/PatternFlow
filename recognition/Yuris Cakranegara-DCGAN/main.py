from dataset_loader import ImageDatasetLoader
from model import OasisDCGAN
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

def main():
    # Replace the train_dataset_dir with the actual directory of the training dataset
    train_dataset_dir = "../../../keras_png_slices_data/keras_png_slices_train/"

    # Define where to store the resulting images
    result_dir = "result_images/"
    os.makedirs(result_dir, exist_ok=True)

    epochs = 100
    batch_size = 64
    img_shape = (256, 256, 1)

    # Load dataset
    print("Loading dataset ...")
    loader = ImageDatasetLoader(train_dataset_dir, img_shape)
    X_train = loader.load_data().astype(np.float32)

    # Reshape and Rescale Images for DCGAN
    # Generator will use tanh activation function for the last layer, 
    # so we want to reshape X_train to be within -1 to 1 limits.
    print("Preprocessing dataset ...")
    X_train = X_train/255
    X_train = X_train.reshape((-1, 256, 256, 1)) * 2. - 1.

    # Generate and store real images for comparison
    img_dir = result_dir + "real_images.png"
    print("Storing real images for comparison to: " + img_dir)
    _fig, a = plt.subplots(3,3, figsize=(10,10))
    images = X_train[:9]
    for i in range(len(images)):
        a[i//3][i%3].set_xticks([])
        a[i//3][i%3].set_yticks([])
        a[i//3][i%3].imshow(images[i][:,:,0], cmap="gray")
    plt.savefig(img_dir)

    # Create and train the model
    model = OasisDCGAN(result_dir=result_dir)
    print("Training the model ...")
    model.train(batch_size, epochs, X_train)

    # Create and store generator loss plot
    print("Plotting generator loss ...")
    plt.clf()
    plt.title('Generator Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(range(1, epochs+1), model.generator_loss)
    plt.savefig(result_dir + "generator_loss.png")

    # Create and store discriminator loss plot
    print("Plotting discriminator loss ...")
    plt.clf()
    plt.title('Discriminator Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(range(1, epochs+1), model.discriminator_loss)
    plt.savefig(result_dir + "discriminator_loss")

    # Calculate SSIM
    noise = tf.random.normal(shape=[9, 100])
    images = model.generator(noise)
    img2 = tf.convert_to_tensor(X_train[0])
    img2 = tf.cast(img2, dtype=tf.float32)

    max_ssim = 0.0
    for img1 in images:
        ssim = tf.image.ssim(img1, img2, max_val=1)
        if ssim > max_ssim: 
            max_ssim = ssim
    tf.print("Structural Similarity:", max_ssim)

if __name__ == "__main__":
    main()
