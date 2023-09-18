from dataset import DataLoader
from modules import Trainer
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt

'''
Plot the original and reconstructed graphs in one line.
'''
def show_subplot(original, reconstructed):
    plt.subplot(1, 2, 1)
    plt.imshow(original.squeeze() + 0.5)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed.squeeze() + 0.5)
    plt.title("Reconstructed")
    plt.axis("off")

    plt.show()

'''
Use trained model to predict and plot 10 predicted images based on tested dataset.
Calculate the structured similarity(SSIM) for each real and reconstructed image pairs.
Input:
        trained_vqvae_model: trained tensorflow.keras model
        test_ds: dataset with numpy array type
'''
def plot(trained_vqvae_model, test_ds):
    idx = np.random.choice(len(test_ds), 10)
    test_images = test_ds[idx]
    reconstructions_test = trained_vqvae_model.predict(test_images)
    for i in range(10):
        tf.keras.preprocessing.image.save_img('./save_image/' + str(i) + '.jpg', reconstructions_test[i])
    for test_image, reconstructed_image in zip(test_images, reconstructions_test):
        show_subplot(test_image, reconstructed_image)
        test_image = tf.convert_to_tensor(test_image)
        pre_image = tf.convert_to_tensor(reconstructed_image)
        test_image = tf.expand_dims(test_image, axis=0)
        pre_image = tf.expand_dims(pre_image, axis=0)
        test_image = tf.image.convert_image_dtype(test_image, tf.float32)
        pre_image = tf.image.convert_image_dtype(pre_image, tf.float32)
        ssim2 = tf.image.ssim(test_image, pre_image, max_val=1.0, filter_size=11,
                            filter_sigma=1.5, k1=0.01, k2=0.03)
        print(ssim2)

def main():
    train_path = ["./keras_png_slices_data/keras_png_slices_data/keras_png_slices_train", "./keras_png_slices_data/keras_png_slices_data/keras_png_slices_seg_train"]
    test_path = ["./keras_png_slices_data/keras_png_slices_data/keras_png_slices_test", "./keras_png_slices_data/keras_png_slices_data/keras_png_slices_test"]
    checkpoint_path = './saved_model/my_model'

    # Load the data
    data_loader = DataLoader()
    train_ds = data_loader.fetch_data(train_path)
    test_ds = data_loader.fetch_data(test_path)
    train_ds_variance = data_loader.get_variance(train_ds)
    test_ds_preprocessed = data_loader.preprocessing(test_ds)
    trained_model = Trainer(img_shape=(256, 256, 1), latent_dim=30, num_embeddings=128, variance=train_ds_variance)
    trained_model.load_weights(checkpoint_path)
    predicted = trained_model.vq_vae.predict(test_ds_preprocessed)

    # Calculate the average ssim of all test images.
    ssim2_total = tf.zeros([1, 1], dtype=tf.float32)
    for test_image, reconstructed_image in zip(test_ds_preprocessed, predicted):
        test_image = tf.convert_to_tensor(test_image)
        pre_image = tf.convert_to_tensor(reconstructed_image)
        test_image = tf.expand_dims(test_image, axis=0)
        pre_image = tf.expand_dims(pre_image, axis=0)
        test_image = tf.image.convert_image_dtype(test_image, tf.float32)
        pre_image = tf.image.convert_image_dtype(pre_image, tf.float32)
        ssim2 = tf.image.ssim(test_image, pre_image, max_val=1.0, filter_size=11,
                            filter_sigma=1.5, k1=0.01, k2=0.03)
        ssim2_total = ssim2 + ssim2_total

    print(ssim2_total/predicted.shape[0])
    plot(trained_model.vq_vae, test_ds_preprocessed)
    
