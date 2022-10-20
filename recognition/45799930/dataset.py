import glob
import tensorflow as tf
from sklearn.utils import shuffle
from math import floor

"""
Create a class to download and sort the ISIC data set. We shall download the training, test and validation data as well 
as all their truth data. 
"""


class DataSet:

    def __init__(self):
        self.validate = None
        self.testing = None
        self.training = None
        self.download_dataset()
        self.image_shape = (256, 256)

    def download_dataset(self):
        """
        This sets up the datasets we need for training, testing or validating. We need both the dataset and the truth
        sets so that we know what it should be after processing. They have been combined in a multidimensional
        tensor.

        :return:

        bool: True if all data sets and their respective truth data sets are the same size, false otherwise.

        """
        # get all the image paths for the different sets - training testing and validation. Then sort these in order so
        # that they are in the same order (truth and initial).
        training_truth_filenames = sorted(glob.glob('./ISIC-2017_Training_Part1_GroundTruth/*.png'))
        training_filenames = sorted(glob.glob('./ISIC-2017_Training_Data/*.jpg'))

        train, truth = shuffle((training_filenames, training_truth_filenames))
        length = len(train)
        training_truth_filenames = truth[:floor(length * 0.8)]
        training_filenames = train[:floor(length * 0.8)]
        testing_truth_filenames = truth[floor(length * 0.8):floor(length * 0.9)]
        testing_filenames = train[floor(length * 0.8):floor(length * 0.9)]
        validate_truth_filenames = truth[floor(length * 0.9):]
        validate_filenames = train[floor(length * 0.9):]

        # convert this into tensorflow array
        self.training = tf.data.Dataset.from_tensor_slices((training_filenames, training_truth_filenames))
        self.testing = tf.data.Dataset.from_tensor_slices((testing_filenames, testing_truth_filenames))
        self.validate = tf.data.Dataset.from_tensor_slices((validate_filenames, validate_truth_filenames))

        self.training = self.training.map(self.pre_process)
        self.testing = self.testing.map(self.pre_process)
        self.validate = self.validate.map(self.pre_process)

        if len(training_truth_filenames) != len(training_filenames) or len(testing_filenames) != len(
                testing_truth_filenames) or len(validate_truth_filenames) != len(validate_filenames):
            return False
        return True

    def pre_process(self, image, truth_image):
        """
        Need to preprocess the data as all we have right now is a location of the image and the truth image. Do this by
        reading the file, decoding the jpeg or png respectively. We check to ensure that all the images are the same
        size. Then cast them to make sure in the same form I cast it to a float.

        :param image: the path to the image.
        :param truth_image: the path to the truth image
        :return: the processed image and truth image.
        """
        image = tf.io.read_file(image)
        # todo: do i need to change the channels? 0 is the number used in the jpeg
        image = tf.io.decode_jpeg(image, channels=0)
        image = tf.image.resize(image, (256, 256))
        image = tf.cast(image, tf.float32) / 255.

        truth_image = tf.io.read_file(truth_image)
        # todo: do i need to change the channels? 0 is the number used in the jpeg
        truth_image = tf.io.decode_png(truth_image, channels=0)
        truth_image = tf.image.resize(truth_image, (256, 256))
        truth_image = tf.cast(truth_image, tf.float32) / 255.
        return image, truth_image

    def split_data(self, data, truths):
        """

        :param training_data:
        :return:
        """
        length = len(data)
        train_len = length * 0.8
        val_test_len = length * 0.1
        data, truths = tf.random.shuffle(data, truths)

        # train, test, val = data.
