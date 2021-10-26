"""
The following code will take two folders as inputs namely the data folder and
the groundtruth folder, and split these into training, testing and
validation.


├── data
└── groundtruth

Convert to:

├── Train
│   ├── Images
│   └── Groundtruth
│
├── Validation
│   ├── Images
│   └── Groundtruth
│
└── Test
    ├── Images
    └── Groundtruth
"""

import sys
import os

def process_data_folders(argv):
    if len(argv) != 3: # We need two arguments
        print("Usage: python3 data_preprocess.py <path to data folder> <path to groundtruth> <path_to_save_data>")
        sys.exit()

    path_to_data = argv[0]
    path_to_groundtruth = argv[1]
    path_to_save_data = argv[2]

    data_files = [f for f in os.listdir(path_to_data) if f.endswith(".jpg")]
    num_files = len(data_files)

    # The groundtruth files are named similarly to the data_files.
    # ISIC_<Number>.jpg
    # ISIC_<Number>_segmentation.png
    # Split data into train, test, validation

    training_split = int(num_files*0.6)
    validation_split = training_split + int(num_files*0.1)
    testing_split = validation_split + int(num_files*0.3)

    training = data_files[0:training_split]
    validation = data_files[training_split:validation_split]
    testing = data_files[validation_split:]

#/home/tannishpage/Documents/COMP3710_DATA/ISIC2018_Task1-2_Training_Input_x2/
#/home/tannishpage/Documents/COMP3710_DATA/ISIC2018_Task1_Training_GroundTruth_x2/
process_data_folders(sys.argv[1:])
