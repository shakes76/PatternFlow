"""
MODEL HYPERPARAMETERS and data import paths
"""

import os

# data
BATCH_SIZE = 128
IMAGE_SIZE = 240  # We'll resize input images to this size
NUM_CLASS = 2
INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 1)

# patches
PATCH_SIZE = 16
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2

# transformer-econder
PROJECTION_DIM = 64
NUM_HEADS = 4
NUM_ENCODER_LAYERS = 6

# mlp head
MLP_HEAD_UNITS = [256]  # Size of the dense layers of the final classifier

# model
VALIDATION_SPLIT = 0.1
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
NUM_EPOCHS = 25
DROPOUTS = {"mha": 0.2, "encoder_mlp": 0.2, "mlp_head": 0.5}

# for image augmentation
NUM_AUG_SAMPLES = 10000

# data import paths
parent_directory = os.getcwd()
path_training = os.path.join(parent_directory, r"AD_NC_square\training")
path_validation = os.path.join(parent_directory, r"AD_NC_square\validation")
path_test = os.path.join(parent_directory, r"AD_NC_square\test")

