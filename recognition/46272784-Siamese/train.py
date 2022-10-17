# This file contains the source code for training, validating, testing and saving my model
import os
import sys
sys.path.insert(1, os.getcwd())
from modules import makeCNN, makeSiamese, loss
import tensorflow as tf
from tensorflow import keras
from tensorflow import train
from tqdm import tqdm
import time

def getOptimizer():
    return keras.optimizers.Adam(1e-4)

def saveOption(optimizer, siamese):
    checkpoint_dir = os.path.join(os.getcwd(), "Siamese_ckeckpoint")
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = train.Checkpoint(optimizer=optimizer,
                                  net=siamese)
    return checkpoint_prefix, checkpoint

