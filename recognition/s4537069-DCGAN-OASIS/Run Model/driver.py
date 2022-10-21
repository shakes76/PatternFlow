
# Import processing and model files to call in main
# Import path to call path to dataset, sys to run file.
import processing
import model
import sys
from pathlib import Path 

# Main function for driver file
# Runs functions from model and processing files which take commandline input of path and epoch.
# python driver.py path epochs

def main(args):
    test_data, train_data, images = processing.pre_processing_data(args[1])
    model.train_model(train_data, int(args[2]), images)

if __name__ == "__main__":
    main(sys.argv)