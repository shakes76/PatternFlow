import glob
import tensorflow as tf
import matplotlib.pyplot as plt

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
        training_truth_filenames = tf.data.Dataset.from_tensor_slices(training_truth_filenames)
        training_filenames = tf.data.Dataset.from_tensor_slices(training_filenames)
        testing_truth_filenames = tf.data.Dataset.from_tensor_slices(testing_truth_filenames)
        testing_filenames = tf.data.Dataset.from_tensor_slices(testing_filenames)
        validate_truth_filenames = tf.data.Dataset.from_tensor_slices(validate_truth_filenames)
        validate_filenames = tf.data.Dataset.from_tensor_slices(validate_filenames)

        # process the data so that we can change the information from
        training_truth = training_truth_filenames.map(self.pre_process_truth)
        training = training_filenames.map(self.pre_process_image)
        testing_truth = testing_truth_filenames.map(self.pre_process_truth)
        testing = testing_filenames.map(self.pre_process_image)
        validate_truth = validate_truth_filenames.map(self.pre_process_truth)
        validate = validate_filenames.map(self.pre_process_image)

        # Let's just check to make sure that the truth is the same length as their corresponding dataset
        if len(training) != len(training_truth) or len(testing) != len(
                testing_truth) or len(validate) != len(validate_truth):
            return False

        self.training = tf.data.Dataset.from_tensor_slices((training, training_truth))
        self.testing = tf.data.Dataset.from_tensor_slices((testing, testing_truth))
        self.validate = tf.data.Dataset.from_tensor_slices((validate, validate_truth))
        return True

    def pre_process_image(self, image):
        """
        Need to preprocess the data as all we have right now is a location of the image. Do this by reading the file,
        decoding the jpeg. We check to ensure that all the images are the same size. Then cast them to make sure in the
        same form I cast it to a float.

        :param image: the path to the image
        :return: the processed image
        """
        image = tf.io.read_file(image)
        # todo: do i need to change the channels? 0 is the number used in the jpeg
        image = tf.io.decode_jpeg(image, channels=0)
        image = tf.image.resize(image, (256, 256))
        image = tf.cast(image, tf.float32) / 255.

        return image

    def pre_process_truth(self, truth_image):
        """
        Need to preprocess the data as all we have right now is a location of the truth image. Do this by reading the
        file, decoding the png. We check to ensure that all the images are the same size. Then cast them to make sure
        in the same form I cast it to a float.

        :param truth_image: the path to the truth_image
        :return: the processed truth image
        """
        truth_image = tf.io.read_file(truth_image)
        # todo: do i need to change the channels? 0 is the number used in the jpeg
        truth_image = tf.io.decode_png(truth_image, channels=0)
        truth_image = tf.image.resize(truth_image, (256, 256))
        truth_image = tf.cast(truth_image, tf.float32) / 255.

        return truth_image
