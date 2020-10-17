import tensorflow as tf
import glob
import os
from recognition.s4436194_oasis_dcgan.data_helper import Dataset

DATA_TRAIN_DIR = "keras_png_slices_data/keras_png_slices_data/keras_png_slices_seg_train"
DATA_TEST_DIR = "keras_png_slices_data/keras_png_slices_data/keras_png_slices_seg_test"
DATA_VALIDATE_DIR = "keras_png_slices_data/keras_png_slices_data/keras_png_slices_seg_validate"


def train_dcgan():
    """Method for training the dcgan on the OASIS MRI images"""

    dataset = Dataset(glob.glob(f"{DATA_TRAIN_DIR}/*.png"))


def test_dcgan():
    pass
