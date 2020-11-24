"""
    File name : driver_test_script.py
    Author : Arun Harish Balasubramonian
    Student Number : 44340326
    Description : Test script where the argument is parsed and actions
                  are evaluated. Possible option includes model training and
                  image generation.
"""

import sys
from data_loader import DataLoader
from trainer import TrainLoader
from generator_loader import GeneratorLoader
import tensorflow as tf


# Mapping all the possible options and its functionality
POSSIBLE_OPTIONS = {
    "--train-dataset" : lambda x, parser: parser.set_training_path(x),
    "--train-epoch" : lambda x, parser: parser.set_training_epoch(x),
    "--train-image-size" : lambda x, parser : parser.set_training_size(x),
    "--generate-image" : lambda parser : parser.set_load_model(True)
}

"""
    Argument Parser that parses the command line argument. If the generate-image option is
    set then does not attempt to train the model, even if other options are present.
    See the README.md test script for more information
"""
class ArgumentParser():
    def __init__(self):
        self.where_training_path = None
        self.total_epoch = None
        self.train_length = None
        # Whether the load mode or --generate-image is set
        self.load_mode = False
    
    # For parsing the training epoch given
    def set_training_epoch(self, value):
        self.total_epoch = int(value)
    
    # For parsing where the input dataset for training comes from
    def set_training_path(self, value):
        self.where_training_path = value
    
    # To parse the total image dataset to be used
    def set_training_size(self, value):
        self.train_length = int(value)
    
    # Set whether the load mode must be set to true or false
    def set_load_model(self, value):
        self.load_mode = value

    # Parses the argument and looks for match with accepted command line arguments
    def parse(self, arguments):
        # Omitting the first argument of filename
        for i in range(1, len(arguments)):
            argument = arguments[i].split("=")
            option = argument[0]
            value = None if len(argument) == 1 else argument[1]
            for current_option in POSSIBLE_OPTIONS:
                if option == current_option:
                    # Calls the function corresponding the options matched
                    if not value is None:
                        POSSIBLE_OPTIONS[current_option](value, self)
                    else:
                        # This can only reach here if the option --generate-image
                        POSSIBLE_OPTIONS[current_option](self)

    # Gives the traning path
    def get_training_path(self):
        return self.where_training_path

    # Gives the training image size to be used for training
    def get_training_size(self):
        return self.train_length

    # Gives the training epoch set
    def get_training_epoch(self):
        return self.total_epoch

    # Indicates whether the load mode is set
    def is_load_mode(self):
        return self.load_mode

if __name__ == "__main__":
    # Here there are three main objects
    # To parse the given arguments
    argument_parser = ArgumentParser()
    # To load the training dataset
    dataset_loader = DataLoader()
    # To train the model 
    trainset_loader = TrainLoader(dataset_loader, argument_parser)

    # Try training the dataset from the source provided in the argument
    # with the settings mentioned in options
    try:
        argument_parser.parse(sys.argv)
        # If in load mode then only load the model and generate the image
        # from random noise
        if argument_parser.is_load_mode():
            generator_loader = GeneratorLoader()
            # Use the loader to generate the image
            generator_loader.generate()
        else:
            # Here then train the model and save its resulting image and the model
            dataset_loader.load(argument_parser)
            # Attempt to use any GPU
            with tf.device("/device:GPU:0"):
                trainset_loader.train()
    except Exception as a:
        print("ERROR during parsing: Make sure you call generate mode only after generating a model")