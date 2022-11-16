"""
parameters.py

Hyperparameters for the visual transformer.

Author: Joshua Wang (Student No. 46965611)
Date Created: 17 Oct 2022
"""
# Hyperparameters
IMAGE_SIZE = 128
PATCH_SIZE = 8
BATCH_SIZE = 32
PROJECTION_DIM = 512 # Depth of MLP blocks
LEARNING_RATE = 0.0005
ATTENTION_HEADS = 5
DROPOUT_RATE = 0.2
TRANSFORMER_LAYERS = 5 # Number of transformer encoder blocks
WEIGHT_DECAY = 0.0001
EPOCHS = 10
MLP_HEAD_UNITS = [256, 128]
DATA_LOAD_PATH = "C:/AD_NC"
MODEL_SAVE_PATH = "C:/model/vision_transformer"

##### AUTOMATICALLY CALCULATED #####
INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
HIDDEN_UNITS = [PROJECTION_DIM * 2, PROJECTION_DIM]
NUM_PATCHES = int((IMAGE_SIZE/PATCH_SIZE) ** 2)