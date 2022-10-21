import os

import cv2
import keras
import numpy as np


class DataGenerator(keras.utils.Sequence):
    def __init__(self, ids, path, batch_size=10, img_size=128):
        """Initialization"""
        self.ids = ids
        self.path = path
        self.batch_size = batch_size  # Number of training samples at one time
        self.img_size = img_size  # size of the pics
        self.on_epoch_end()

    def __load__(self, id_name):
        """load each image from local file path"""
        # file path
        image_path = os.path.join(self.path, "images", id_name) + ".jpg"  # train image
        mask_path = os.path.join(self.path, "masks", id_name) + "_segmentation.png"  # segmentation image

        # Read the original image and the correctly segmented image separately
        # Since out image is in GrayScale form but we need image in 3 channel form
        train_image = cv2.imread(image_path, 1)  # Load color images
        train_image = cv2.resize(train_image, (self.img_size, self.img_size))  # adjust size
        mask = np.zeros((self.img_size, self.img_size, 1))  # 128 * 128 * 1
        mask_image = cv2.imread(mask_path, -1)  # cv2.IMREAD_UNCHANGED
        mask_image = cv2.resize(mask_image, (self.img_size, self.img_size))
        mask_image = np.expand_dims(mask_image, axis=-1)
        mask = np.maximum(mask, mask_image)  # Select maximum

        # normalize the images in range [0, 1]
        train_image = train_image / 255.0
        mask = mask / 255.0

        return train_image, mask

    def __getitem__(self, index):
        """Generate one batch of data : Returns i'th batch"""
        if (index + 1) * self.batch_size > len(self.ids):
            self.batch_size = len(self.ids) - index * self.batch_size

        # Take the batch of directories
        files_batch = self.ids[index * self.batch_size: (index + 1) * self.batch_size]

        image = []
        mask = []

        # pass each directory name to __load__() method, where we extract both image and masks and perform operations.
        for id_name in files_batch:
            _img, _mask = self.__load__(id_name)
            image.append(_img)
            mask.append(_mask)

        image = np.array(image)
        mask = np.array(mask)

        return image, mask  # Return array

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        pass

    def __len__(self):
        """Denotes the number of batches per epoch : Returns number of batches"""
        return int(np.ceil(len(self.ids) / float(self.batch_size)))
