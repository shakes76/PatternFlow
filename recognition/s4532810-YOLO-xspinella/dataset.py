import gdown
from zipfile import ZipFile
import os
import PIL
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms as T
import utils
import pandas as pd

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
        #self.Resize_Images()

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
            print(f"============= Directory {i} =================")
            j = 0
            for file in files:
                path = os.path.join(directory_list[i], file)
                self.Resize_Image(path)
                j += 1
            i += 1

    def Resize_Image(self, path):
        img = Image.open(path)
        check_arr = [np.array(img).shape[0], np.array(img).shape[1]]
        if not(check_arr == [512, 512]):
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
        c_x = min_left + (w/2)
        c_y = min_up + (h/2)
        # bounding box params are normalised amd returned
        return [c_x/512, c_y/512, w/512, h/512]
         
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
    
    def Find_Class_From_CSV(self, img_fp: str, csv_fp: str):
        """
        Find the class (0:!melanoma, 1:melanoma) of the given
        filename, by matching the id from the filename to
        the id in the row of the csv
        :param img_fp: filepath of the gnd truth image of interest
        :param csv_fp: filepath of corresponding csv file for classification
        """
        ### Find image id from the fp ###
        # remove directories from fp string
        last_slash_idx = img_fp.rfind('/')
        img_fp = img_fp[last_slash_idx+1:]
        # extract img id
        dot_idx = img_fp.rfind('_')
        img_id = img_fp[0:dot_idx]

        ### Find the classification from given csv ###
        # find row corresponding to the img_id 
        img_df = pd.read_csv(csv_fp)
        img_arr = img_df.values
        id_row = np.NaN
        i = 0
        for row in img_arr:
            if row[0] == img_id:
                id_row = i
                break
            i += 1
        if id_row == np.NaN:
            raise LookupError("The image ID was not found in CSV file")
        # return the melanoma classification (0:!melanoma, 1:melanoma)
        return int(img_arr[id_row][1])

    def Create_YOLO_Labels(self):
        """
        Creates a corresponding txt file 'label' for each image in 
        each dataset, and places them in label folder. txt file name 
        is the same as corresponding file. i.e. if the image is 
        1234.jpg, the txt label file will be 1234.txt. The format
        of the txt file will be <class> <c_x> <c_y> <w> <h>
        """
        # Define directories to loop thru
        dataset_list = ["ISIC_data/extract_files/Test", 
                        "ISIC_data/extract_files/Train", 
                        "ISIC_data/extract_files/Validate"]
        curr_dir_list = ["Test", "Train", "validate"]
        # loop thru directories
        for dataset in dataset_list:
            dir_items = os.listdir(dataset)
            csv_path = ""
            gnd_truth_dir = ""
            # find gnd truth dir and csv classification file
            for item in dir_items:
                if item.endswith(".csv"):
                    csv_path = os.path.join(dataset, item)
                elif item.endswith("_PNG"):
                    gnd_truth_dir = os.path.join(dataset, item)
            # At this stage we have the path for the classification CSV
            # and the folder which contains the folder which contains all the
            # gnd truth mask segmentations. So next we extract this inner folder:
            inner_folder = os.listdir(gnd_truth_dir)[0]
            gnd_truth_dir = os.path.join(gnd_truth_dir, inner_folder)
            # Now we have the actual path to all the gnd truth masks,
            # so we can extract all the masks:
            gnd_truth_masks = os.listdir(gnd_truth_dir)
            for mask_path in gnd_truth_masks:
                # Find the YOLO label corresponding to this mask
                self.Get_YOLO_Label(mask_path, csv_path)
                # create txt file and save label to it
                pass
    
    def Get_YOLO_Label(self, mask_path: str, csv_path: str):
        """
        :param mask_path: path to mask segmentation of image to produce
                            label for
        :param csv_path: path to csv which contains the melanoma classification
                            for this image id
        :return: The YOLO-format label for this image id:
                    normalised([class, C_x, C_y, w, h])
        """
        ### Find the box specs and class spec ###
        normalised_box_spec = self.Mask_To_Box(mask_path)
        # remember 0:!melanoma, 1:melanoma
        melanoma_class = self.Find_Class_From_CSV(mask_path, csv_path)

        ### Concatenate in correct order ###
        normalised_box_spec.insert(0, float(melanoma_class))
        return normalised_box_spec 

def Setup_Data():
    ### Initialise dataloader - this implicitly deletes unwanted files and downloads/extracts/resizes datasets ###
    dataloader = DataLoader()

    ### Verify helper functions are working ###
    gnd_truth = dataloader.train_truth_PNG_ex + "/ISIC-2017_Training_Part1_GroundTruth/ISIC_0000002_segmentation.png"
    train_img = dataloader.train_data_ex + "/ISIC-2017_Training_Data/ISIC_0000002.jpg"
    # Verify that the bounding box code is working for an isolated case:
    print(dataloader.Mask_To_Box(gnd_truth))
    # Verify that img class lookup function is working for an isolated case:
    print(dataloader.Find_Class_From_CSV(gnd_truth, dataloader.train_truth_gold))
    # Verify that label creation function works
    label = dataloader.Get_YOLO_Label(gnd_truth, dataloader.train_truth_gold)
    # Verify that draw function is working
    utils.Draw_Box(gnd_truth, label, "misc_tests/box_test_truth.png")
    utils.Draw_Box(train_img, label, "misc_tests/box_test_img.png")

    ### generate a txt file for each img which specifies bounding box, and class of object ###
    # note that -> 0:melanoma, 1:!melanoma
    # dataloader.Create_YOLO_Labels()
     
    ### move to directories as required by yolov5 ###

    ### convert raw data/gnd truths into format we want ###

Setup_Data()

# TODO: Questions to ask:

#