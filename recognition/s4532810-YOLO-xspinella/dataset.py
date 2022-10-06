import gdown
from zipfile import ZipFile
import os

class DataLoader():
    """ 
    class used to load all relevant data and preprocess/arrange as
    required
    """
    def __init__(self):
        """
        """
        ### Define locations and names for zip/download files ###
        # Datasets
        self.train_data = "ISIC_data/zip_files/Train/Train_data.zip"
        self.test_data = "ISIC_data/zip_files/Test/Test_data.zip"
        self.validation_data = "ISIC_data/zip_files/Validate/Valid_data.zip"
        # Ground Truths - unsure why there is 3 for each dataset?
        self.train_truth_PNG = "ISIC_data/zip_files/Train/Train_Truth_PNG.zip"
        self.train_truth_JSON = "ISIC_data/zip_files/Train/Train_Truth_JSON.zip"
        self.train_truth_gold = "ISIC_data/extract_files/Train/Train_Truth_gold.csv"
        self.test_truth_PNG = "ISIC_data/zip_files/Test/Test_Truth_PNG.zip"
        self.test_truth_JSON = "ISIC_data/zip_files/Test/Test_Truth_JSON.zip"
        self.test_truth_gold = "ISIC_data/extract_files/Test/Test_Truth_gold.csv"
        self.valid_truth_PNG = "ISIC_data/zip_files/Validate/Valid_Truth_PNG.zip"
        self.valid_truth_JSON = "ISIC_data/zip_files/Validate/Valid_Truth_JSON.zip"
        self.valid_truth_gold = "ISIC_data/extract_files/Validate/Valid_Truth_gold.csv"

        ### Define download urls ###
        # Datasets:
        self.train_data_url = "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Data.zip"
        self.test_data_url = "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Test_v2_Data.zip"
        self.validation_data_url = "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Validation_Data.zip"
        # Ground Truths:
        self.train_truth_PNG_url = "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Part1_GroundTruth.zip"
        self.train_truth_JSON_url = "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Part2_GroundTruth.zip"
        self.train_truth_gold_url = "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Part3_GroundTruth.csv"
        self.test_truth_PNG_url = "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Test_v2_Part1_GroundTruth.zip"
        self.test_truth_JSON_url = "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Test_v2_Part2_GroundTruth.zip"
        self.test_truth_gold_url = "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Test_v2_Part3_GroundTruth.csv"
        self.valid_truth_PNG_url = "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Validation_Part1_GroundTruth.zip"
        self.valid_truth_JSON_url = "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Validation_Part2_GroundTruth.zip"
        self.valid_truth_gold_url = "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Validation_Part3_GroundTruth.csv"

        ### Download all the zip/download files ###
        self.Download_Zips()

        ### Define locations for extracted files ###     
        # TODO: Figure out how the test/train/validation sets need to be structured in the file
        # Datasets
        self.train_data_ex = "ISIC_data/extract_files/Train/Train_data"
        self.test_data_ex = "ISIC_data/extract_files/Test/Test_data"
        self.validation_data_ex = "ISIC_data/extract_files/Validate/Valid_data"
        # Ground Truths - note that 'gold' gnd truth is a csv file so no need to extract - unsure why there is 3 for each dataset?
        self.train_truth_PNG_ex = "ISIC_data/extract_files/Train/Train_Truth_PNG"
        self.train_truth_JSON_ex = "ISIC_data/extract_files/Train/Train_Truth_JSON"
        self.test_truth_PNG_ex = "ISIC_data/extract_files/Test/Test_Truth_PNG"
        self.test_truth_JSON_ex = "ISIC_data/extract_files/Test/Test_Truth_JSON"
        self.valid_truth_PNG_ex = "ISIC_data/extract_files/Validate/Valid_Truth_PNG"
        self.valid_truth_JSON_ex = "ISIC_data/extract_files/Validate/Valid_Truth_JSON"

        ### Extract all zip files into required directories ###
        self.Extract_Zips()

    def Download_Zips(self):
        """
        Downloads the zip files into allocated folders, 
        if they don't already exist
        """
        # Download datasets:
        if not(os.path.exists(self.train_data)):
            print("Downloading train dataset")
            gdown.download(self.train_data_url, self.train_data, quiet=False)
        if not(os.path.exists(self.test_data)):
            print("Downloading test dataset")
            gdown.download(self.test_data_url, self.test_data, quiet=False)
        if not(os.path.exists(self.validation_data)):
            print("Downloading validation dataset")
            gdown.download(self.validation_data_url, self.validation_data, quiet=False)
        # Download train set ground truths:
        if not(os.path.exists(self.train_truth_PNG)):
            print("Downloading train PNG dataset")
            gdown.download(self.train_truth_PNG_url, self.train_truth_PNG, quiet=False)
        if not(os.path.exists(self.train_truth_JSON)):
            print("Downloading train JSON dataset")
            gdown.download(self.train_truth_JSON_url, self.train_truth_JSON, quiet=False)
        if not(os.path.exists(self.train_truth_gold)):
            print("Downloading train gold dataset")
            gdown.download(self.train_truth_gold_url, self.train_truth_gold, quiet=False)
        # Download test set ground truths:
        if not(os.path.exists(self.test_truth_PNG)):
            print("Downloading test PNG dataset")
            gdown.download(self.test_truth_PNG_url, self.test_truth_PNG, quiet=False)
        if not(os.path.exists(self.test_truth_JSON)):
            print("Downloading test JSON dataset")
            gdown.download(self.test_truth_JSON_url, self.test_truth_JSON, quiet=False)
        if not(os.path.exists(self.test_truth_gold)):
            print("Downloading test gold dataset")
            gdown.download(self.test_truth_gold_url, self.test_truth_gold, quiet=False)
        # Download validation set ground truths:
        if not(os.path.exists(self.valid_truth_PNG)):
            print("Downloading validation PNG dataset")
            gdown.download(self.valid_truth_PNG_url, self.valid_truth_PNG, quiet=False)
        if not(os.path.exists(self.valid_truth_JSON)):
            print("Downloading valid JSON dataset")
            gdown.download(self.valid_truth_JSON_url, self.valid_truth_JSON, quiet=False)
        if not(os.path.exists(self.valid_truth_gold)):
            print("Downloading valid gold dataset")
            gdown.download(self.valid_truth_gold_url, self.valid_truth_gold, quiet=False)

    def Extract_Zips(self):
        """
        Extracts all Zip files into the directories specified in constructor,
        if they aren't already extracted.
        """
        zip_list = [self.train_data, self.test_data, self.validation_data,
                    self.train_truth_PNG, self.train_truth_JSON,
                    self.test_truth_PNG, self.test_truth_JSON,
                    self.valid_truth_PNG, self.valid_truth_JSON]
        location_list = [self.train_data_ex, self.test_data_ex, self.validation_data_ex,
                        self.train_truth_PNG_ex, self.train_truth_JSON_ex,
                        self.test_truth_PNG_ex, self.test_truth_JSON_ex,
                        self.valid_truth_PNG_ex, self.valid_truth_JSON_ex]
        i = 0
        while i < len(zip_list):
            if not(os.path.exists(location_list[i])):
                print(f"extracting item {i}")
                with ZipFile(zip_list[i], "r") as zipobj:
                    zipobj.extractall(location_list[i])
            i += 1


def debug():
    ### load/save/extract all raw data and gnd truths ###
    dataloader = DataLoader()

    ### convert raw data/gnd truths into format we want ###
    # Convert segment mask into bounding box x, y, w, h - isnt x, y supposed to be specified relative to each grid cell? how does this work
    # make 4 classes(?): {0:[!m, !s_k], 1:[m, !s_k], 2:[!m, s_k], 3:[m, s_k]}? so one num can be used to specify each case
    # Combine bounding box data with class data 
    # make separate txt file for each image, containing combined data
    ### move to directories as required by yolov5 ###
    # use https://medium.com/mlearning-ai/training-yolov5-custom-dataset-with-ease-e4f6272148ad to organise label (txt) files and images
    ### make yaml file for training specification ###
    ### train ###
debug()

# TODO: Questions to ask:
# 1. am I just using yolo to draw box around the lesions? or is it this + classification
# 2. what is the "superpixels"
# 3. what are all the three different downloadsfor the ground truths
# 4. it would appear that: truth_PNG is the segmentations, and truth_gold is the classification
#       Would it be correct to say that I only need these two (wtf is the JSON)?
# 5. am I ok to use this YOLOv5 library? 
# 6. what is modules.py for (general library?)  