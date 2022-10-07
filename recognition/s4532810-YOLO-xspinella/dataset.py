import gdown
from zipfile import ZipFile
import os
import PIL
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms as T

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
        self.train_truth_gold = "ISIC_data/extract_files/Train/Train_Truth_gold.csv"
        self.test_truth_PNG = "ISIC_data/zip_files/Test/Test_Truth_PNG.zip"
        self.test_truth_gold = "ISIC_data/extract_files/Test/Test_Truth_gold.csv"
        self.valid_truth_PNG = "ISIC_data/zip_files/Validate/Valid_Truth_PNG.zip"
        self.valid_truth_gold = "ISIC_data/extract_files/Validate/Valid_Truth_gold.csv"

        ### Define download urls ###
        # Datasets:
        self.train_data_url = "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Data.zip"
        self.test_data_url = "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Test_v2_Data.zip"
        self.validation_data_url = "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Validation_Data.zip"
        # Ground Truths:
        self.train_truth_PNG_url = "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Part1_GroundTruth.zip"
        self.train_truth_gold_url = "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Part3_GroundTruth.csv"
        self.test_truth_PNG_url = "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Test_v2_Part1_GroundTruth.zip"
        self.test_truth_gold_url = "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Test_v2_Part3_GroundTruth.csv"
        self.valid_truth_PNG_url = "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Validation_Part1_GroundTruth.zip"
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
        self.test_truth_PNG_ex = "ISIC_data/extract_files/Test/Test_Truth_PNG"
        self.valid_truth_PNG_ex = "ISIC_data/extract_files/Validate/Valid_Truth_PNG"

        ### Extract all zip files into required directories ###
        self.Extract_Zips()

        ### Delete unwanted files ###
        self.Delete_Unwanted_Files()

        ### resize all images to 512x512 ###
        self.Resize_Images()

    def Resize_Images(self):
        """
        Resizes all images and segmentation masks to 256x256
        """
        ### Make a list of lists, where each nested list contains a list of filenames in a directory ###
        directory_list = [self.train_data_ex + "/ISIC-2017_Training_Data",
                            self.train_truth_PNG_ex + "/ISIC-2017_Training_Part1_GroundTruth",
                            self.test_data_ex + "/ISIC-2017_Test_v2_Data", 
                            self.test_truth_PNG_ex + "/ISIC-2017_Test_v2_Part1_GroundTruth",
                            self.validation_data_ex + "/ISIC-2017_Validation_Data",
                            self.valid_truth_PNG_ex + "/ISIC-2017_Validation_Part1_GroundTruth"]
        files_list = []
        for directory in directory_list:
            files_list.append(os.listdir(directory))
        ### loop thru each directory, and resize each image (resize function saves over original) ###
        i = 0
        for files in files_list:
            for file in files:
                path = os.path.join(directory_list[i], file)
                self.Resize_Image(path)
            i += 1

    def Resize_Image(self, path):
        img = Image.open(path)
        transform = T.Resize((512, 512))

        # apply the transform on the input image, and save over the old one
        img = transform(img)
        img.save(path)

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
        if not(os.path.exists(self.train_truth_gold)):
            print("Downloading train gold dataset")
            gdown.download(self.train_truth_gold_url, self.train_truth_gold, quiet=False)
        # Download test set ground truths:
        if not(os.path.exists(self.test_truth_PNG)):
            print("Downloading test PNG dataset")
            gdown.download(self.test_truth_PNG_url, self.test_truth_PNG, quiet=False)
        if not(os.path.exists(self.test_truth_gold)):
            print("Downloading test gold dataset")
            gdown.download(self.test_truth_gold_url, self.test_truth_gold, quiet=False)
        # Download validation set ground truths:
        if not(os.path.exists(self.valid_truth_PNG)):
            print("Downloading validation PNG dataset")
            gdown.download(self.valid_truth_PNG_url, self.valid_truth_PNG, quiet=False)
        if not(os.path.exists(self.valid_truth_gold)):
            print("Downloading valid gold dataset")
            gdown.download(self.valid_truth_gold_url, self.valid_truth_gold, quiet=False)

    def Extract_Zips(self):
        """
        Extracts all Zip files into the directories specified in constructor,
        if they aren't already extracted.
        """
        zip_list = [self.train_data, self.test_data, self.validation_data,
                    self.train_truth_PNG, self.test_truth_PNG,
                    self.valid_truth_PNG]
        location_list = [self.train_data_ex, self.test_data_ex, self.validation_data_ex,
                        self.train_truth_PNG_ex, self.test_truth_PNG_ex,
                        self.valid_truth_PNG_ex]
        i = 0
        while i < len(zip_list):
            if not(os.path.exists(location_list[i])):
                print(f"extracting item {i}")
                with ZipFile(zip_list[i], "r") as zipobj:
                    zipobj.extractall(location_list[i])
            i += 1

    def Mask_To_Box(self, img_fp: str):
        """
        Converts given segment mask into bounding box specification:
        x, y, w, h
        :param img: filepath to segment mask of one of the lesions
        :return: Bounding box definition as [centre_x, centre_y, width, height]
        """
        # Open image and convert to array
        img = Image.open(img_fp)
        img_arr = np.array(img)

        # define vars for pointing out the bounds of the box
        min_left = np.inf
        max_right = -np.inf
        min_up = np.inf
        max_down = -np.inf
        # found the bounds of the box:
        for i in range(0, len(img_arr)):        # Rows
            for j in range(0, len(img_arr[0])): # Cols
                if img_arr[i][j] > 0:
                    min_left = min(min_left, j)
                    max_right = max(max_right, j)
                    min_up = min(min_up, i)
                    max_down = max(max_down, i)
        # redefine as centre_x, centre_y, width, height
        w = max_right - min_left
        h = max_down - min_up
        C_x = min_left + (w/2)
        C_y = min_up + (h/2)

        return [C_x, C_y, w, h]

    def Draw_Box(self, img_fp: str, box_spec: list):
        """
        Draws the specified box on the given image
        :param img_fp: filepath to the image
        :param box_spec: box size and location specification as:
                            [centre_x, centre_y, width, height]
        """
        ### open image with cv2 and save image size ###
        img = cv2.imread()
        height, width, _ = img.shape

        ### redefine box location ###
         
    def Delete_Unwanted_Files(self):
        """
        Deletes files that are not needed. These are the superpixel images in
        the data folders.
        """
        ### Make a list of lists, where each nested list contains a list of filenames in a directory ###
        directory_list = [self.train_data_ex + "/ISIC-2017_Training_Data", 
                            self.test_data_ex + "/ISIC-2017_Test_v2_Data", 
                            self.validation_data_ex + "/ISIC-2017_Validation_Data"]
        files_list = []
        for directory in directory_list:
            files_list.append(os.listdir(directory))
        ### loop thru each directory, and delete files ending with superpixels.png ###
        i = 0
        for files in files_list:
            for file in files:
                if file.endswith("superpixels.png"):
                    path = os.path.join(directory_list[i], file)
                    os.remove(path)
            i += 1
        

def debug():
    ### load/save/extract all raw data and gnd truths ###
    dataloader = DataLoader()

    ### convert raw data/gnd truths into format we want ###
    # resize all images to be the same
    # Convert segment mask into bounding box x, y, w, h - isnt x, y supposed to be specified relative to each grid cell? how does this work
    # box = dataloader.Mask_To_Box("{}/ISIC-2017_Training_Part1_GroundTruth/ISIC_0000000_segmentation.png".format(dataloader.train_truth_PNG_ex)) # Define folder path"
    # print(box)

    # make 4 classes(?): {0:[!m, !s_k], 1:[m, !s_k], 2:[!m, s_k], 3:[m, s_k]}? so one num can be used to specify each case
    #       just have two classes melonoma/sk or !melonoma/sk
    # Combine bounding box data with class data 
    # make separate txt file for each image, containing combined data
    ### move to directories as required by yolov5 ###
    # use https://medium.com/mlearning-ai/training-yolov5-custom-dataset-with-ease-e4f6272148ad to organise label (txt) files and images
    ### make yaml file for training specification ###
    ### train ###
debug()

# TODO: Questions to ask:
# 1. am I just using yolo to draw box around the lesions? or is it this + classification yes
# 2. what is the "superpixels"? delete them all
# 3. what are all the three different downloads for the ground truths - dont use JSON files
# 4. it would appear that: truth_PNG is the segmentations, and truth_gold is the classification
#       Would it be correct to say that I only need these two (wtf is the JSON)?
# 5. am I ok to use this YOLOv5 library? 
# 6. what is modules.py for (general library?)  yes

#