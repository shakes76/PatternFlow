from glob import glob
import dataset as dp
import modules as iu
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

dataset_path = "D:/Data/Mylecture/COMP3710/lab_report/data/"

seg_test_path = sorted(glob(dataset_path + "ISIC-2017_Test_v2_Part1_GroundTruth/.png"))
seg_train_path = sorted(glob(dataset_path + "ISIC-2017_Training_Part1_GroundTruth/*.png"))
seg_val_path = sorted(glob(dataset_path + "ISIC-2017_Validation_Part1_GroundTruth/*.png"))
test_path = sorted(glob(dataset_path + "ISIC-2017_Test_v2_Data/*_???????.jpg"))
train_path = sorted(glob(dataset_path + "ISIC-2017_Training_Data/*_???????.jpg"))
val_path = sorted(glob(dataset_path + "ISIC-2017_Validation_Data/*_???????.jpg"))

# create the dataset
train_ds = tf.data.Dataset.from_tensor_slices((train_path, seg_train_path))
val_ds = tf.data.Dataset.from_tensor_slices((val_path, seg_val_path))
test_ds = tf.data.Dataset.from_tensor_slices((test_path, seg_test_path))

# load the dataset
train_ds = train_ds.map(dp.process_image)
val_ds = val_ds.map(dp.process_image)
test_ds = test_ds.map(dp.process_image)

test, seg_test = next(iter(test_ds.batch(len(test_path))))