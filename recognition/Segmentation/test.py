import tensorflow as tf
import glob
import pathlib
import os 

dataset_path = './datasets'
dataset_path = pathlib.Path(__file__).parent.absolute() / 'datasets/OASIS'

#Load data
train_images = sorted(glob.glob(str(dataset_path / "keras_png_slices_train/*.png")))
train_masks = sorted(glob.glob(str(dataset_path / "keras_png_slices_seg_train/*.png")))

test_images = sorted(glob.glob(str(dataset_path / "keras_png_slices_test/*.png")))
test_masks = sorted(glob.glob(str(dataset_path / "keras_png_slices_seg_test/*.png")))

val_images = sorted(glob.glob(str(dataset_path / "keras_png_slices_validate/*.png")))
val_masks = sorted(glob.glob(str(dataset_path / "keras_png_slices_seg_validate/*.png")))

print(test_images)