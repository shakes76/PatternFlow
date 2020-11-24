"""
    File name : data_loader.py
    Author : Arun Harish Balasubramonian
    Student Number : 44340326
    Description : A common module that loads the given dataset onto memory 
                  for model training.
"""

from os import listdir, path
import matplotlib.image as mpimg

"""
    Reponsible for loading the data from the source given in the argument parser.
    Stores the image and sources them upon request by the Trainer.py.
    Normalises the dataset to be in the range [-1, 1] for further use in the training
    process.
"""
class DataLoader():
    def __init__(self):
        # Initial value
        self.train_set = []
    
    # Called by the driver script to load the images given by the user
    def load(self, parser):
        # Gets the training path and source it from absolute path
        which_file = parser.get_training_path()
        which_file_abs_path = path.abspath(which_file)
        # Load all the images here through listdir assuming
        # only the image dataset is present in the directory
        for i in listdir(which_file_abs_path):
            # Getting all the image as numerical values in the directory.
            source = "{}/{}".format(which_file_abs_path, i)
            image_read = mpimg.imread(source)
            # Normalising value for further use in trainer
            # since the activation function is tanh used by the generator
            image_read = image_read * 2 - 1.0
            self.train_set.append(
                image_read
            )
    # Retrieves the entire dataset as requested by the trainer
    def get_dataset(self):
        return self.train_set