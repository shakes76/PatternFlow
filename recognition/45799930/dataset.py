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

    def download_dataset(self):
        """
        impotant information here..
        :return:
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


data = DataSet()
print(type(data))
