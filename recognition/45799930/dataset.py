import glob
import tensorflow as tf
import matplotlib.pyplot as plt

"""
Create a class to download and sort the ISIC data set. We shall download the training, test and validation data as well 
as all their truth data. 
"""


class DataSet:

    def __init__(self):
        self.validate_filenames = None
        self.validate_truth_filenames = None
        self.testing_filenames = None
        self.testing_truth_filenames = None
        self.training_filenames = None
        self.training_truth_filenames = None
        self.download_dataset()
        self.image_shape = (256, 256)

    def download_dataset(self):
        """
        This sets up the datasets we need for training, testing or validating. We need both the dataset and the truth
        sets so that we know what it should be after processing.

        :return:

        bool: True if all data sets and their respective truth data sets are the same size, false otherwise.

        """

        # get all the image paths for the different sets - training testing and validation. Then sort these in order so
        # that they are in the same order (truth and initial).
        training_truth_filenames = sorted(glob.glob('./ISIC-2017_Training_Part1_GroundTruth/*.png'))
        training_filenames = sorted(glob.glob('./ISIC-2017_Training_Data/*.jpg'))
        testing_truth_filenames = sorted(glob.glob('./ISIC-2017_Test_v2_Part1_GroundTruth/*.png'))
        testing_filenames = sorted(glob.glob('./ISIC-2017_Test_v2_Data/*.jpg'))
        validate_truth_filenames = sorted(glob.glob('./ISIC-2017_Validation_Part1_GroundTruth/*.png'))
        validate_filenames = sorted(glob.glob('./ISIC-2017_Validation_Data/*.jpg'))

        # convert this into tensorflow array
        self.training_truth_filenames = tf.data.Dataset.from_tensor_slices(training_truth_filenames)
        self.training_filenames = tf.data.Dataset.from_tensor_slices(training_filenames)
        self.testing_truth_filenames = tf.data.Dataset.from_tensor_slices(testing_truth_filenames)
        self.testing_filenames = tf.data.Dataset.from_tensor_slices(testing_filenames)
        self.validate_truth_filenames = tf.data.Dataset.from_tensor_slices(validate_truth_filenames)
        self.validate_filenames = tf.data.Dataset.from_tensor_slices(validate_filenames)
        # Let's just check to make sure that the truth is the same length as their corresponding dataset
        if len(self.training_filenames) != len(self.training_truth_filenames) or len(self.testing_filenames) != len(
                self.testing_filenames) or len(self.validate_filenames) != len(self.validate_truth_filenames):
            return False
        return True

    def pre_process(self, image, truth_image):
        """
        Need to preprocess the data as all we have right now is a location of the image and truth
        image. Do this by reading the file, decoding the jpeg or png respectively. We check to
        ensure that all the images are the same size. Then cast them to make sure in the same form
        I cast both of them to make it easier for me.

        :param image: the path to the image
        :param truth_image: the path to the ground truth image.
        :return: a tuple containing the processed image and ground-truth image.
        """
        image = tf.io.read_file(image)
        # todo: do i need to change the chanels? 0 is the number used in the jpeg
        image = tf.io.decode_jpeg(image, channels=0)
        image = tf.image.resize(image, (256, 256))
        image = tf.cast(image, tf.float32) / 255.

        truth_image = tf.io.read_file(truth_image)
        # todo: do i need to change the chanels? 0 is the number used in the jpeg
        truth_image = tf.io.decode_png(truth_image, channels=0)
        truth_image = tf.image.resize(truth_image, (256, 256))
        truth_image = tf.cast(truth_image, tf.float32) / 255.

        return image, truth_image
