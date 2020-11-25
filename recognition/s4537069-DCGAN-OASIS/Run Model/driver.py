import processing
import model
import sys
from pathlib import Path 

def main(args):
    test_data, train_data, images = processing.pre_processing_data(args[1])
    model.train_model(train_data, int(args[2]), images)

if __name__ == "__main__":
    main(sys.argv)