import sys
from data_loader import DataLoader
from trainer import TrainLoader
from generator_loader import GeneratorLoader
import tensorflow as tf

POSSIBLE_OPTIONS = {
    "--train-dataset" : lambda x, parser: parser.set_training_path(x),
    "--train-epoch" : lambda x, parser: parser.set_training_epoch(x),
    "--train-image-size" : lambda x, parser : parser.set_training_size(x),
    "--generate-image" : lambda parser : parser.set_load_model(True)
}

class ArgumentParser():
    def __init__(self):
        self.where_training_path = None
        self.total_epoch = None
        self.train_length = None
        self.load_mode = False
    
    def set_training_epoch(self, value):
        self.total_epoch = int(value)
    
    def set_training_path(self, value):
        self.where_training_path = value
    
    def set_training_size(self, value):
        self.train_length = int(value)

    def set_load_model(self, value):
        self.load_mode = value

    def parse(self, arguments):
        for i in range(1, len(arguments)):
            argument = arguments[i].split("=")
            option = argument[0]
            value = None if len(argument) == 1 else argument[1]
            for current_option in POSSIBLE_OPTIONS:
                if option == current_option:
                    if not value is None:
                        POSSIBLE_OPTIONS[current_option](value, self)
                    else:

                        POSSIBLE_OPTIONS[current_option](self)

    def get_training_path(self):
        return self.where_training_path

    def get_training_size(self):
        return self.train_length

    def get_training_epoch(self):
        return self.total_epoch

    def is_load_mode(self):
        return self.load_mode

if __name__ == "__main__":
 
    argument_parser = ArgumentParser()
    dataset_loader = DataLoader()
    trainset_loader = TrainLoader(dataset_loader, argument_parser)
    
    try:
        argument_parser.parse(sys.argv)

        if argument_parser.is_load_mode():
            generator_loader = GeneratorLoader()
            # Use the loader to generate the image
            generator_loader.generate()
        else:
            dataset_loader.load(argument_parser)
            with tf.device("/device:GPU:0"):
                trainset_loader.train()
    except:
        print("ERROR during parsing")