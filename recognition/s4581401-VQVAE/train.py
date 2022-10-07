import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from dataset import *
from modules import *

val_split = 0.2
img_height = 256
img_width = 256
batch_size = 32
latent_dim = 32
embedding_num = 256


PATH = os.getcwd()
print(PATH)

train_path = PATH + "/ADNI_AD_NC_2D/AD_NC/train"
test_path = PATH + "/ADNI_AD_NC_2D/AD_NC/test"
train_ds = load_train_data_no_val(train_path, img_height, img_width, batch_size)
test_ds = load_test_data(test_path, img_height, img_width, batch_size)

#Need to calculate variance in the training data.
vqvae_model = VQVAEModel(img_shape = (256, 256, 3), embedding_num = 256, embedding_dim= latent_dim, beta = 0.25, data_variance=0.05)
print(vqvae_model.get_encoder().summary())
print(vqvae_model.get_vq().output)
print(vqvae_model.get_decoder().summary())