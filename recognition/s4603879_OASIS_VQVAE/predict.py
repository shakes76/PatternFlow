from tabnanny import check
from dataset import DataLoader
from modules import Trainer
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras as keras


def main():
    train_path = ["./keras_png_slices_data/keras_png_slices_data/keras_png_slices_train", "./keras_png_slices_data/keras_png_slices_data/keras_png_slices_seg_train"]
    test_path = ["./keras_png_slices_data/keras_png_slices_data/keras_png_slices_test", "./keras_png_slices_data/keras_png_slices_data/keras_png_slices_test"]
    checkpoint_path = './saved_model/my_model'
    data_loader = DataLoader()
    train_ds = data_loader.fetch_data(train_path)
    test_ds = data_loader.fetch_data(test_path)
    train_ds_variance = data_loader.get_variance(train_ds)
    test_ds_preprocessed = data_loader.preprocessing(test_ds)
    trained_model = Trainer(img_shape=(256, 256, 1), latent_dim=30, num_embeddings=128, variance=train_ds_variance)
    trained_model.load_weights(checkpoint_path)
    predicted = trained_model.vq_vae.predict(test_ds_preprocessed)
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
    
