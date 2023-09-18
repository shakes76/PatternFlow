from dataset import DataLoader
from modules import Trainer
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt


'''
Retrieve and data and create a vqvae model and train it using 50 epochs.
The trained wights will be saved to /saved_model. The losses of each epoch will be plotted after the training.
Input:
        train_path: the paths that contain the training datasets.
'''
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
    history = vqvae_trainer.fit(train_ds_preprocessed)
    print(history.history.keys())
    plt.plot(history.history['loss'])
    plt.plot(history.history['reconstruction_loss'])
    plt.plot(history.history['vqvae_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['loss', 'reconstruction_loss', 'vqvae_loss'], loc='upper right')
    plt.show()


def main():
    train_path = ["./keras_png_slices_data/keras_png_slices_data/keras_png_slices_train", "./keras_png_slices_data/keras_png_slices_data/keras_png_slices_seg_train"]
    test_path = ["./keras_png_slices_data/keras_png_slices_data/keras_png_slices_test", "./keras_png_slices_data/keras_png_slices_data/keras_png_slices_test"]
    train(train_path)


if __name__ == '__main__':
    main()
    