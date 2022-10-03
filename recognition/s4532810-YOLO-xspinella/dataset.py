import gdown
from zipfile import ZipFile
import os

### Download and save zipped training, testing and validation sets ###
# Define zip file location and name
Train_output = "ISIC_data/Train_data.zip"
Test_output = "ISIC_data/Test_data.zip"
Validation_output = "ISIC_data/Valid_data.zip"

if not(os.path.exists(Train_output) and os.path.exists(Test_output) and os.path.exists(Validation_output)):
    # Define download urls
    Train_url = "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Data.zip"
    Test_url = "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Test_v2_Data.zip"
    Validation_url = "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Validation_Data.zip"

    # Download and save zipped sets, as specified above
    gdown.download(Train_url, Train_output, quiet=True)
    gdown.download(Test_url, Test_output, quiet=True)
    gdown.download(Validation_url, Validation_output, quiet=True)

# TODO:
# 1. Figure out how the test/train/validation sets need to be structured in the file
# 2. use ZipFile to 
# with ZipFile("celeba_gan/data.zip", "r") as zipobj:
#     zipobj.extractall("celeba_gan")


# TODO: Questions to ask:
# 1. am I just using yolo to draw box around the lesions? or is it this + 