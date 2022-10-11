from google.colab import drive
import numpy as np
import tensorflow as tf
from os import listdir
from os.path import isfile, join
from PIL import Image
import zipfile
import matplotlib.pyplot as plt
import numpy as np

googleCollab = False
if (googleCollab):
    drive.mount('/content/gdrive', force_remount=True);
    zipped_images = zipfile.ZipFile('/content/gdrive/MyDrive/Colab Notebooks/OASISProcessed.zip')
else:
    zipped_images = zipfile.ZipFile('/content/gdrive/MyDrive/Colab Notebooks/OASISProcessed.zip')
images = []
zipped_images.extractall()

parent_dir = "./keras_png_slices_data"
train_path = parent_dir + "/keras_png_slices_train"
test_path = parent_dir + "/keras_png_slices_test"
val_path = parent_dir + "/keras_png_slices_validate"
train_images = [np.array(Image.open(join(train_path, f))) for f in listdir(train_path) if isfile(join(train_path, f))]
test_images = [np.array(Image.open(join(test_path, f))) for f in listdir(test_path) if isfile(join(test_path, f))]
validate_images = [np.array(Image.open(join(val_path, f))) for f in listdir(val_path) if isfile(join(val_path, f))]

