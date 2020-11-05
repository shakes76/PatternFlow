from dataset_loader import ImageDatasetLoader
from model import OasisDCGAN
import matplotlib.pyplot as plt
import os

def main():
    train_dataset_dir = "../../../keras_png_slices_data/keras_png_slices_train/"
    result_dir = "result_images/"
    os.makedirs(result_dir, exist_ok=True)

    epochs = 250
    batch_size = 64
    img_shape = (256, 256, 1)

    # Load dataset
    loader = ImageDatasetLoader(train_dataset_dir, img_shape)
    X_train = loader.load_data()

    # Reshape and Rescale Images for DCGAN
    # Generator will use tanh activation function for the last layer, 
    # so we want to reshape X_train to be within -1 to 1 limits.
    X_train = X_train/255
    X_train = X_train.reshape((-1, 256, 256, 1)) * 2. - 1.

    # Generate and store real images for comparison
    _fig, a = plt.subplots(3,3, figsize=(10,10))
    images = X_train[:9]
    for i in range(len(images)):
        a[i//3][i%3].set_xticks([])
        a[i//3][i%3].set_yticks([])
        a[i//3][i%3].imshow(images[i], cmap="gray")
    plt.savefig(result_dir + "real_images.png")

    # Create and train the model
    model = OasisDCGAN(result_dir=result_dir)
    model.train(batch_size, epochs, X_train)

    # Crate and store generator loss plot
    plt.title('Generator Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(range(1, epochs+1), model.generator_loss)
    plt.savefig("generator_loss.png")

    # Crate and store discriminator loss plot
    plt.title('Discriminator Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(range(1, epochs+1), model.discriminator_loss)
    plt.savefig("discriminator_loss")


if __name__ == "__main__":
    main()