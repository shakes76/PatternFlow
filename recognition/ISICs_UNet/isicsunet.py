"""
Class for components of model used for ISICs UNet recognition problem.

Created by Christopher Bailey (45576430) for COMP3710 Report.

Data is extracted from the Preprocessed ISIC 2018 Melanoma Dermoscopy Dataset
provided on course blackboard.
"""


import tensorflow as tf


class IsicsUnet:
    def __init__(self):
        self.train_ds = None
        self.val_ds = None

    def load_data(self):
        """
        Downloads and prepares the data set for use in the model

        Based on code from https://www.tensorflow.org/tutorials/load_data/images
        and code from COMP3710-demo-code.ipynb from Guest Lecture.
        """

        # download data
        dataset_url = 'https://cloudstor.aarnet.edu.au/sender/download.php?token=505165ed-736e-4fc5-8183-755722949d34&files_ids=10012238'
        file_name = 'isic_2018.zip'
        data_dir = tf.keras.utils.get_file(fname=file_name, origin=dataset_url, extract=True)
        print("Data dir:", data_dir)

        # load all filenames
        import glob
        data_dir = data_dir.replace(file_name, '')
        image_filenames = glob.glob(data_dir + 'ISIC2018_Task1-2_Training_Input_x2/*.jpg')
        mask_filenames = [f.replace('ISIC2018_Task1-2_Training_Input_x2',
                                    'ISIC2018_Task1_Training_GroundTruth_x2')
                          .replace('jpg', 'png') for f in image_filenames]

        # expected number of images is 2594
        image_count = len(image_filenames)
        mask_count = len(mask_filenames)
        print("Image count:", image_count, "Mask count:", mask_count)

        # split the dataset, 80% train 20% validate
        val_size = int(image_count * 0.2)
        val_images = image_filenames[:val_size]
        val_masks = mask_filenames[:val_size]
        train_images = image_filenames[val_size:]
        train_masks = mask_filenames[val_size:]
        print("Size of training set:", len(train_images), len(train_masks))
        print("Size of validation set:", len(val_images), len(val_masks))

        # create TensorFlow Dataset and shuffle it
        self.train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_masks))
        self.val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_masks))

        self.train_ds = self.train_ds.shuffle(len(train_images))
        self.val_ds = self.val_ds.shuffle(len(val_images))

