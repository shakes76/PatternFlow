import glob
import tensorflow as tf

"""
Create a class to download and sort the ISIC data set. We shall download the training, test and validation data as well 
as all their truth data. 
"""


class DataSet:

    def __int__(self):
        self.training_set = 0
        self.validation_set = 0
        self.testing_set = 0
        self.download_dataset()

    def download_dataset(self):
        """
        impotant information here..
        :return:
        """

        # get all the image paths for the different sets - training testing and validation. Then sort these in order so
        # that they are in the same order (truth and initial).
        training_truth_filenames = sorted(glob.glob('ISIC-2017_Training_Part1_GroundTruth/*.jpg'))
        training_filenames = sorted(glob.glob('ISIC-2017_Training_Data/*.jpg'))
        testing_truth_filenames = sorted(glob.glob('ISIC-2017_Test_v2_Part1_GroundTruth/*.jpg'))
        testing_filenames = sorted(glob.glob('ISIC-2017_Test_v2_Data/*.jpg'))
        validate_truth_filenames = sorted(glob.glob('ISIC-2017_Validation_Part1_GroundTruth/*.jpg'))
        validate_filenames = sorted(glob.glob('ISIC-2017_Validation_Data/*.jpg'))

        # convert this into tensorflow array
        training_truth_filenames = tf.data.Dataset.from_tensor_slices(training_truth_filenames)
        training_filenames = tf.data.Dataset.from_tensor_slices(training_filenames)
        testing_truth_filenames = tf.data.Dataset.from_tensor_slices(testing_truth_filenames)
        testing_filenames = tf.data.Dataset.from_tensor_slices(testing_filenames)
        validate_truth_filenames = tf.data.Dataset.from_tensor_slices(validate_truth_filenames)
        validate_filenames = tf.data.Dataset.from_tensor_slices(validate_filenames)

        


