import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

num_classes = 100
input_shape = (64, 64, 3)

learning_rate = 0.001
weight_decay = 0.0001
# batch_size = 64
batch_size = 16
num_epochs = 50
dropout_rate = 0.2
image_size = 64  # We'll resize input images to this size.
patch_size = 2  # Size of the patches to be extract from the input images.
num_patches = (image_size // patch_size) ** 2  # Size of the data array.
latent_dim = 256  # Size of the latent array.
projection_dim = 256  # Embedding size of each element in the data and latent arrays.
num_heads = 8  # Number of Transformer heads.
ffn_units = [
    projection_dim,
    projection_dim,
]  # Size of the Transformer Feedforward network.
num_transformer_blocks = 4
num_iterations = 2  # Repetitions of the cross-attention and Transformer modules.
classifier_units = [
    projection_dim,
    num_classes,
]  # Size of the Feedforward network of the final classifier.



# text files describing which is train and which is validation data, setup in datasetup.py
train_file_path = 'train.txt'
test_file_path = 'validation.txt'

#paths where class folders are
train_data_path = 'data/resize/'
test_data_path = train_data_path

train_num = 11352
test_num = 4768
iterations_per_epoch = int(train_num / batch_size)
test_iterations = int(test_num / batch_size) + 1
warm_iterations = iterations_per_epoch
