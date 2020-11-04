"""
Class for components of model used for ISICs UNet recognition problem.

Created by Christopher Bailey (45576430) for COMP3710 Report.

Data is extracted from the Preprocessed ISIC 2018 Melanoma Dermoscopy Dataset
provided on course blackboard.

Segments of code in this file are based on code from COMP3710-demo-code.ipynb
from COMP3710 Guest Lecture and code from TensorFlow tutorial pages.
"""

import tensorflow as tf

IMAGE_WIDTH = 512
IMAGE_HEIGHT = 384
IMAGE_CHANNELS = 3

MASK_WIDTH = 324
MASK_HEIGHT = 196
MASK_CHANNELS = 1

class IsicsUnet:
    def __init__(self):
        self.train_ds = None
        self.val_ds = None
        self.model = None
        self.BATCH_SIZE = 1


    @staticmethod
    def map_fn(image, mask):
        """
        Helper function to map dataset filenames to the actual image data arrays

        Based on code from COMP3710-demo-code.ipynb from Guest Lecture.
        """

        # load image
        img = tf.io.read_file(image)
        img = tf.image.decode_jpeg(img, channels=IMAGE_CHANNELS)
        img = tf.image.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))  # resize all images to min size

        # normalize image to [0,1]
        img = tf.cast(img, tf.float32) / 255.0

        # load mask
        m = tf.io.read_file(mask)
        m = tf.image.decode_png(m, channels=MASK_CHANNELS)
        m = tf.image.resize(m, (MASK_WIDTH, MASK_HEIGHT))  # resize all masks to min size

        # normalize mask to [0,1]
        m = tf.cast(m, tf.float32) / 255.0

        # do we need to one-hot encode the mask? theres only one channel anyway?
        #m = tf.keras.utils.to_categorical(m)

        return img, m

    def visualise_loaded_data(self):
        """
        Helper function to visualise loaded image and mask data

        Based on code from COMP3710-demo-code.ipynb from Guest Lecture.
        """
        image_batch, mask_batch = next(iter(self.train_ds.batch(3)))
        print("Image batch shape:", image_batch.numpy().shape)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 10))
        for i in range(3):
            plt.subplot(3, 2, 2 * i + 1)
            plt.imshow(image_batch[i])
            plt.title("Input Image")
            plt.axis('off')
            plt.subplot(3, 2, 2 * i + 2)
            plt.imshow(mask_batch[i])
            plt.title("True mask")
            plt.axis('off')
        plt.show()

    def visualise_segmentation(self):
        """
        Helper function to visualise loaded image and mask data

        Based on code from COMP3710-demo-code.ipynb from Guest Lecture.
        """
        pass

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
                          .replace('.jpg', '_segmentation.png') for f in image_filenames]

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

        # map filenames to data arrays
        self.train_ds = self.train_ds.map(IsicsUnet.map_fn)
        self.val_ds = self.val_ds.map(IsicsUnet.map_fn)

        for image, mask in self.train_ds.take(1):
            print('Image shape:', image.numpy().shape)
            print('Mask shape:', mask.numpy().shape)

        ## batch data set
        #self.train_ds = self.train_ds.batch(self.batch_size, drop_remainder=True)
        #self.val_ds = self.train_ds.batch(self.batch_size, drop_remainder=True)

    def build_model(self):
        """
        Build the U-Net model using TensorFlow functional API
        """

        # encoder/downsampling
        input_size = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)
        inputs = tf.keras.Input(input_size)
        conv1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(inputs)
        conv1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(conv1)
        pool1 = tf.keras.layers.MaxPool2D((2, 2))(conv1)

        conv2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(pool1)
        conv2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(conv2)
        pool2 = tf.keras.layers.MaxPool2D((2, 2))(conv2)

        conv3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu')(pool2)
        conv3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu')(conv3)
        pool3 = tf.keras.layers.MaxPool2D((2, 2))(conv3)

        conv4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu')(pool3)
        conv4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu')(conv4)
        pool4 = tf.keras.layers.MaxPool2D((2, 2))(conv4)

        # bridge/bottleneck/shared layer
        conv5 = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu')(pool4)
        conv5 = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu')(conv5)

        # decoder/upsampling
        up6 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv5)
        conv4 = tf.keras.layers.Cropping2D(cropping=((4, 4), (4, 4)))(conv4)
        up6 = tf.keras.layers.concatenate([conv4, up6])
        conv6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu')(up6)
        conv6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu')(conv6)

        up7 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv6)
        conv3 = tf.keras.layers.Cropping2D(cropping=((16, 17), (16, 17)))(conv3)
        up7 = tf.keras.layers.concatenate([conv3, up7])
        conv7 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu')(up7)
        conv7 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu')(conv7)

        up8 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv7)
        conv2 = tf.keras.layers.Cropping2D(cropping=((41, 41), (41, 41)))(conv2)
        up8 = tf.keras.layers.concatenate([conv2, up8])
        conv8 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(up8)
        conv8 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(conv8)

        up9 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv8)
        conv1 = tf.keras.layers.Cropping2D(cropping=((90, 90), (90, 90)))(conv1)
        up9 = tf.keras.layers.concatenate([conv1,up9])
        conv9 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(up9)
        conv9 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(conv9)

        # segmentation (output) layer
        outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='softmax')(conv9)

        self.model = tf.keras.Model(inputs=inputs,outputs=outputs)

    def train_model(self):
        """
        Train the model
        """
        pass

    def show_predictions(self):
        """
        Perform prediction on validation set and report performance

        Based on code from COMP3710-demo-code.ipynb from Guest Lecture.
        """
        image_batch, mask_batch = next(iter(self.val_ds.batch(3)))
        predictions = self.model.predict(image_batch)
        # TODO: is it required to convert to predicted masks?

        import matplotlib.pyplot as plt
        plt.figure(figsize=(20,10))
        for i in range(3):
            # show base image
            plt.subplot(3,3,3*i+1)
            plt.imshow(image_batch[i])
            plt.axis('off')

            # show true mask
            plt.subplot(3,3,3*i+2)
            plt.imshow(mask_batch[i])
            plt.axis('off')

            # show predicted mask
            plt.subplot(3,3,3*i+3)
            plt.imshow(predictions[i])
            plt.axis('off')

        plt.show()
