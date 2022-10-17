import tensorflow as tf
from matplotlib import image, pyplot
import glob
import numpy as np


class Dataset:
    # Initialise where the data is located (previously downloaded and extracted)
    def __init__(self):
        path = "C:/Users/Georgia Spanevello/.keras/datasets/oasis/keras_png_slices_data"
        self.train_data_path = path + "/keras_png_slices_train"
        self.train_labels_path = path + "/keras_png_slices_seg_train"
        self.test_data_path = path + "/keras_png_slices_test"
        self.test_labels_path = path + "/keras_png_slices_seg_test"
        self.valid_data_path = path + "/keras_png_slices_validate"
        self.valid_labels_path = path + "/keras_png_slices_seg_validate"

    # Load and process the data at the given path
    def load_process_images(self, path):
        images = []

        # Read all png images
        for file in glob.glob(path + "/*.png"):
            images.append(image.imread(file))
        
        # Numpy array to hold all the now numpy array converted images
        data = np.array(images, dtype = np.float32)

        # Residual extraction
        data = (data - np.mean(data)) / np.std(data)
        # Normalise
        data = (data - np.amin(data)) / np.amax(data - np.amin(data))

        return data[:, :, :, np.newaxis]
    
    # Load and process (one-hot encoding) the labels at the given path
    def load_process_labels(self, path):
        images = []

        for file in glob.glob(path + "/*.png"):
            curr_image = image.imread(file)
            one_hot_enc = np.zeros(curr_image.shape)
            for i, value in enumerate(np.unique(curr_image)):
                one_hot_enc[:, :][curr_image == value] = i
            images.append(one_hot_enc)

        labels = np.array(images, dtype = np.uint8)

        one_hot_encs = []
        num_classes = 4
        for i in range(labels.shape[0]):
            curr_image = labels[i]
            one_hot_enc = np.zeros((curr_image.shape[0], curr_image.shape[1], num_classes), dtype = np.uint8)
        
            for i, value in enumerate(np.unique(curr_image)):
                one_hot_enc[:, :, i][curr_image == value] = 1
                one_hot_encs.append(one_hot_enc)

        return np.array(one_hot_encs)


