"""
OASIS Brain Dataset Segmentation with Improved UNet, 
with all labels having a minimum Dice Similarity Coefficient 
of 0.9 on the test set.

@author Dhilan Singh (44348724)

Start Date: 01/11/2020
"""
import tensorflow as tf

import glob

print('Tensorflow Version:', tf.__version__)

# Download the dataset (use the direct link given on the page)
print("> Loading images ...")
# tf.keras.utils.get file downloads a file from a URL if it not already in the cache.
#     origin: Original URL of the file.
#     fname: Name of the file. If an absolute path /path/to/file.txt is specified the 
#            file will be saved at that location (in cache directory).
#            NEEDS FILE EXTENSION TO WORK!!!
#     extract: If true, extracting the file as an Archive, like tar or zip.
#     archive_format: zip, tar, etc...
#     returns: Path to the downloaded file.
dataset_url = "https://cloudstor.aarnet.edu.au/plus/s/n5aZ4XX1WBKp6HZ/download"
data_path = tf.keras.utils.get_file(origin=dataset_url,
                                    fname="keras_png_slices_data.zip",
                                    extract=True,
                                    archive_format="zip")

# Remove the .zip file extension from the data path
data_path_clean = data_path.split('.zip')[0]

# Load filenames into a list in sorted order
train_images = sorted(glob.glob(data_path_clean + "/keras_png_slices_train/*.png"))
train_masks = sorted(glob.glob(data_path_clean +"/keras_png_slices_seg_train/*.png"))
val_images = sorted(glob.glob(data_path_clean + "/keras_png_slices_validate/*.png"))
val_masks = sorted(glob.glob(data_path_clean + "/keras_png_slices_seg_validate/*.png"))
test_images = sorted(glob.glob(data_path_clean + "/keras_png_slices_test/*.png"))
test_masks = sorted(glob.glob(data_path_clean + "/keras_png_slices_seg_test/*.png"))

# Build tf datasets
train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_masks))
val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_masks))
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_masks))

# Make the dataset to be reshuffled each time it is iterated over.
# This is so that we get different batches for each epoch.
# For perfect shuffling, the buffer size needs to be greater than or equal to the size of the dataset.
train_ds = train_ds.shuffle(len(train_images))
val_ds = val_ds.shuffle(len(val_images))
test_ds = test_ds.shuffle(len(test_images))



# End of operation
print('End')
