import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob
import pathlib
from PIL import Image
import zipfile

def main():
    #Get the file location of where the data is stored.
    data_dir = pathlib.Path('H:/Year 3/Sem 2/COMP3710/Report/PatternFlow/recognition/keras_png_slices_data/keras_png_slices_data')
    #The following 5 lines were used to download and unzip the data, these have been commented out for faster debugging.
    #However, they have been left in to show how the files were retreived.
    #data_dir_tf = tf.keras.utils.get_file(origin='https://cloudstor.aarnet.edu.au/plus/s/n5aZ4XX1WBKp6HZ/download',
    #                                      fname='H:/Year 3/Sem 2/COMP3710/Report/PatternFlow/recognition/keras_png_slices_data.zip')

    #with zipfile.ZipFile(data_dir_tf) as zf:
    #    zf.extractall()

    #List the file paths of the data
    test_scans = sorted(glob.glob('keras_png_slices_data/keras_png_slices_test/*.png'))
    test_labels = sorted(glob.glob('keras_png_slices_data/keras_png_slices_seg_test/*.png'))
    train_scans = sorted(glob.glob('keras_png_slices_data/keras_png_slices_train/*.png'))
    train_labels = sorted(glob.glob('keras_png_slices_data/keras_png_slices_seg_train/*.png'))
    val_scans = sorted(glob.glob('keras_png_slices_data/keras_png_slices_validate/*png'))
    val_labels = sorted(glob.glob('keras_png_slices_data/keras_png_slices_seg_validate/*.png'))

    train_dataset = tf.data.Dataset.from_tensor_slices((train_scans, train_labels))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_scans, val_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_scans, test_labels))

    train_dataset = train_dataset.shuffle(len(train_scans))
    val_dataset = val_dataset.shuffle(len(val_scans))
    test_dataset = test_dataset.shuffle(len(test_scans))

    
    
##    test_brain = Image.open(str(test_scans[0]))
##    test_brain = np.asarray(test_brain, dtype=np.uint8)
##    print(test_brain.shape)

if __name__ == "__main__":
    main()
