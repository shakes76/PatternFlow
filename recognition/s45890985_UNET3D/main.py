import nibabel as nib
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from model import *
from tensorflow.keras import datasets, layers, models, callbacks
from tensorflow.keras.utils import Sequence
from pyimgaug3d.augmenters import ImageSegmentationAugmenter
from pyimgaug3d.augmentation import Flip, GridWarp
from pyimgaug3d.utils import to_channels


class NiftiDataGenerator(Sequence):

    def __init__(self, file_names, image_path, to_fit=True,
                 batch_size=1, dim=(256, 256, 128),
                 n_channels=1, n_seg=6, shuffle=True):
        self.file_names = file_names
        self.image_path = image_path
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_seg = n_seg
        self.shuffle = shuffle
        self.indexes = range(len(file_names))

    def __len__(self):
        return int(np.floor(len(self.file_names) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_filenames_temp = [self.file_names[i] for i in indexes]

        img, mask = self.__data_generation(list_filenames_temp)

        return img, mask

    def on_epoch_end(self):
        self.indexes = np.aramge(len(self.file_names))

    def aug_3d_data(img, seg):
        aug = ImageSegmentationAugmenter()
        aug.add_augmentation(Flip(1))
        aug_img, aug_seg = aug([img, seg])
        return aug_img, aug_seg
    
    def load_nifti_files(self, path):
        # find corresponding mask path
        mask_path = path.replace('MRs', 'labels')
        mask_path = mask_path.replace('LFOV', 'SEMANTIC_LFOV')

        # load image as nparray
        img = nib.load(path).get_fdata()
        img = img / 255.0
        mask = nib.load(mask_path).get_fdata()
        mask = to_channels(mask)
        img = img[..., None]

        return img, mask

    def __data_generation(self, list_filenames_temp):
        # create empty arrays to store batches of data
        imgs = np.empty((self.batch_size, *self.dim, self.n_channels))
        masks = np.empty((self.batch_size, *self.dim, self.n_seg), dtype=int)

        # generate data
        for i, name in enumerate(list_filenames_temp):
            imgs[i,], masks[i,] = self.load_nifti_files(os.path.join(self.image_path, name))

        return imgs, masks

def main():
    data_dir = "D:/UQ/2021 Sem 2/COMP3710/Report/HipMRI_study_complete_release_v1/semantic_MRs_anon"
    mri_paths = os.listdir(data_dir)
    split = int(np.ceil(len(mri_paths) * 0.8))
    train_files = mri_paths[:split]
    val_files = mri_paths[split:]

    training_generator = NiftiDataGenerator(train_files, data_dir)
    validation_generator = NiftiDataGenerator(val_files, data_dir)

    model = unet_model()
    epoch_steps = len(train_files)
    epochs = 2
    validation_steps = len(val_files)
    model.fit(training_generator, epochs=epochs, steps_per_epoch=epoch_steps,
              validation_steps=validation_steps, validation_data=validation_generator)
    model.save(os.getcwd())

if __name__ == "__main__":
    main()