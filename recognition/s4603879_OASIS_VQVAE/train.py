from dataset import DataLoader
from modules import Trainer
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras as keras


def train(train_path):
    data_loader = DataLoader()
    train_ds = data_loader.fetch_data(train_path)
    print(train_ds.shape)
    train_ds_preprocessed = data_loader.preprocessing(train_ds)
    train_ds_variance = data_loader.get_variance(train_ds)
    checkpoint_path = './saved_model/my_model'
    # checkpoint_dir = os.path.dirname(checkpoint_path)
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)
    
    vqvae_trainer = Trainer(img_shape=(256, 256, 1), latent_dim=30, num_embeddings=128, variance=train_ds_variance)
    vqvae_trainer.compile(optimizer=keras.optimizers.Adam())
    vqvae_trainer.fit(train_ds_preprocessed, epoch=50, batch_size=128, callback=[cp_callback])


def main():
    train_path = ["./keras_png_slices_data/keras_png_slices_data/keras_png_slices_train", "./keras_png_slices_data/keras_png_slices_data/keras_png_slices_seg_train"]
    test_path = ["./keras_png_slices_data/keras_png_slices_data/keras_png_slices_test", "./keras_png_slices_data/keras_png_slices_data/keras_png_slices_test"]
    data_loader = DataLoader()
    train_ds = data_loader.fetch_data(train_path)
    print(train_ds.shape)
    train_ds_preprocessed = data_loader.preprocessing(train_ds)
    test_ds = data_loader.fetch_data(test_path)
    test_ds_preprocessed = data_loader.preprocessing(test_ds)
    train_ds_variance = data_loader.get_variance(train_ds)
    vqvae_trainer = Trainer(img_shape=(256, 256, 1), latent_dim=30, num_embeddings=128, variance=train_ds_variance)
    vqvae_trainer.compile(optimizer=keras.optimizers.Adam())
    vqvae_trainer.fit(train_ds_preprocessed)


if __name__ == '__main__':
    main()
    