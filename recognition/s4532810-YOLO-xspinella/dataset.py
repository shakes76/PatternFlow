import gdown
from zipfile import ZipFile
import os
import PIL
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms as T
import utils_lib
import pandas as pd
import shutil

class DataLoader():
    """ 
    class used to load all relevant data and preprocess/arrange as
    required
    """
    def __init__(self):
        """
        """
        ### Make all required directories ###
        self.Create_File_Structure()
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

        ### resize all images to 640x640 ###
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
        curr_dir_name = ["Train Data", "Train Masks", "Test Data", "Test Masks",
                            "Validation Data", "Validation Masks"]
        files_list = []
        for directory in directory_list:
            files_list.append(os.listdir(directory))
        ### loop thru each directory, and resize each image (resize function saves over original) ###
        i = 0
        for files in files_list:
            print(f"============= Resize {curr_dir_name[i]} =================")
            j = 0
            for file in files:
                path = os.path.join(directory_list[i], file)
                self.Resize_Image(path)
                j += 1
            i += 1

    def Resize_Image(self, path):
        img = Image.open(path)
        check_arr = [np.array(img).shape[0], np.array(img).shape[1]]
        if not(check_arr == [640, 640]):
            transform = T.Resize((640, 640))
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
    
    def Create_File_Structure(self):
        """
        Creates all directories that are required for functionality of
        this yolo implementation. should be called at the very
        start of the initialiser
        """
        ### Directories to store all raw data while is is being manipulated into yolo format ###
        top_dir = "ISIC_data"
        zip_dir = "ISIC_data/zip_files"
        extract_files = "ISIC_data/extract_files"

        ### directories which zip file downloads are stored in ###
        test_zipdir = "ISIC_data/zip_files/Test"
        train_zipdir = "ISIC_data/zip_files/Train"
        valid_zipdir = "ISIC_data/zip_files/Validate"

        ### directories which zip files are extracted to ###
        test_dir = "ISIC_data/extract_files/Test"
        train_dir = "ISIC_data/extract_files/Train"
        valid_dir = "ISIC_data/extract_files/Validate"

        ### directories to be used by actual yolo alg ###
        images_dir = "yolov5_LC/data/images"
        labels_dir = "yolov5_LC/data/labels"
        train_images_dir = "yolov5_LC/data/images/training"
        valid_images_dir = "yolov5_LC/data/images/validation"
        test_images_dir = "yolov5_LC/data/images/testing"
        train_labels_dir = "yolov5_LC/data/labels/training"
        valid_labels_dir = "yolov5_LC/data/labels/validation"
        test_labels_dir = "yolov5_LC/data/labels/testing"

        ### Output/testing directories ###
        miscell = "misc_tests"
        pred_out = "pred_out"
        test_out = "test_out"

        ### create directories ###
        dirs = [
            top_dir, zip_dir, extract_files,
            test_zipdir, train_zipdir, valid_zipdir,
            test_dir, train_dir, valid_dir,
            images_dir, labels_dir, train_images_dir,
            valid_images_dir, test_images_dir, train_labels_dir,
            valid_labels_dir, test_labels_dir, miscell, 
            pred_out, test_out]
        for dir in dirs:
            if not(os.path.exists(dir)):
                os.mkdir(dir)

    def Create_YOLO_Labels(self, ignore_existing=False):
        """
        Creates a corresponding txt file 'label' for each image in 
        each dataset, and places them in label folder. txt file name 
        is the same as corresponding file. i.e. if the image is 
        1234.jpg, the txt label file will be 1234.txt. The format
        of the txt file will be <class> <c_x> <c_y> <w> <h>
        :param ignore_existing: set to true if you wish to overwrite
                                all existing txt files. when false, 
                                label txt files that already exist will
                                not be overwritten
        """
        # Define directories to loop thru
        dataset_list = ["ISIC_data/extract_files/Test", 
                        "ISIC_data/extract_files/Train", 
                        "ISIC_data/extract_files/Validate"]
        yolo_dir_set = ["yolov5_LC/data/labels/testing",
                        "yolov5_LC/data/labels/training",
                        "yolov5_LC/data/labels/validation"]
        curr_dir_list = ["test", "train", "validate"]
        # loop thru directories
        i = 0
        for dataset in dataset_list:
            print(f"Creating labels for {curr_dir_list[i]} set")
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
                mask_path = os.path.join(gnd_truth_dir, mask_path)
                
                img_id = utils_lib.Get_Gnd_Truth_Img_ID(mask_path)
                path = f"{yolo_dir_set[i]}/{img_id}.txt"
                if not(os.path.exists(path)) or ignore_existing:
                    # Find the YOLO label corresponding to this mask
                    label, img_id = utils_lib.Get_YOLO_Label(mask_path, csv_path)
                    # create txt file and save label to it
                    np.savetxt(path, np.array([label]), fmt='%f')
            i += 1

    def Copy_Images(self, ignore_existing=False):
        """
        Copies the train/Test/Validate images to the required 
        yolov5 folder.
        :param ignore_existing: set to true if you wish to overwrite
                        all existing images. When false, 
                        images that already exist will
                        not be overwritten
        """
        ### Define directories to loop thru ###
        source_list = ["ISIC_data/extract_files/Test/Test_data/ISIC-2017_Test_v2_Data", 
                        "ISIC_data/extract_files/Train/Train_data/ISIC-2017_Training_Data", 
                        "ISIC_data/extract_files/Validate/Valid_data/ISIC-2017_Validation_Data"]
        dest_list = ["yolov5_LC/data/images/testing",
                        "yolov5_LC/data/images/training",
                        "yolov5_LC/data/images/validation"]
        curr_dir_list = ["test", "train", "validate"]

        ### loop through sources and copy contents to destinations ###
        i = 0
        while i < len(source_list):
            images = os.listdir(source_list[i])
            print(f"Copying images from {curr_dir_list[i]} set")
            for img in images:
                dest_path = os.path.join(dest_list[i], img)
                if not(os.path.exists(dest_path)) or ignore_existing:
                    source_path = os.path.join(source_list[i], img)
                    shutil.copy(source_path, dest_list[i])
            i += 1

    def Copy_Configs(self, ignore_existing=False):
        """
        Copies the two required .yaml config files
        from the patternflow to the YOLOv5 git submodule.
        :param ignore_existing: set to true if you wish to overwrite
                                all existing images. When false, 
                                images that already exist will
                                not be overwritten
        """
        source_dir = "config_files"
        configs = ["ISIC_dataset.yaml",
                    "ISIC_test.yaml"]
        destination = "yolov5_LC/data"
        print("Copying .yaml configs")
        for config in configs:
            dest_path = os.path.join(destination, config)
            if not(os.path.exists(dest_path)) or ignore_existing:
                src_path = os.path.join(source_dir, config)
                shutil.copy(src_path, destination)


def Setup_Data():
    """
    Same as Download_Preprocess_Arrange() function in train.py
    exists if the user wants to arrange/preprocess all data without
    executing train.py
    """
    ### Initialise dataloader, this implicitly: 
    #   deletes unwanted files, 
    #   downloads/extracts/resizes datasets 
    #   creates directory structure ###
    dataloader = DataLoader()

    ### Verify helper functions are working ###
    gnd_truth = dataloader.train_truth_PNG_ex + "/ISIC-2017_Training_Part1_GroundTruth/ISIC_0000002_segmentation.png"
    train_img = dataloader.train_data_ex + "/ISIC-2017_Training_Data/ISIC_0000002.jpg"
    # Verify that the bounding box code is working for an isolated case:
    print(f"Test conversion of mask to box specs: Should return '[0.54296875, 0.56640625, 0.6078125, 0.7828125]'\
        {utils_lib.Mask_To_Box(gnd_truth)}")
    # Verify that img class lookup function is working for an isolated case:
    print(f"Test melanoma lookup function. Should return '(1, 'ISIC_0000002')'\n \
        {utils_lib.Find_Class_From_CSV(gnd_truth, dataloader.train_truth_gold)}")
    # Verify that label creation function works
    label, img_id = utils_lib.Get_YOLO_Label(gnd_truth, dataloader.train_truth_gold)
    np.savetxt(f"misc_tests/{img_id}.txt", np.array([label]), fmt='%f')
    # Verify that draw function is working
    utils_lib.Draw_Box(gnd_truth, label, "misc_tests/box_test_truth.png")
    utils_lib.Draw_Box(train_img, label, "misc_tests/box_test_img.png")

    ### generate a txt file for each img which specifies bounding box, and class of object ###
    # note that -> 0:melanoma, 1:!melanoma
    dataloader.Create_YOLO_Labels() 

    # Verify that box draw function from txt file label works
    label_fp = "yolov5_LC/data/labels/training/ISIC_0000002.txt"
    utils_lib.Draw_Box_From_Label(label_fp, train_img, "misc_tests/box_from_label.png")  

    ### Copy images to directories as required by yolov5 ###
    dataloader.Copy_Images()

    ### copy yaml file into correct YOLOv5 directory ###
    dataloader.Copy_Configs()

# Setup_Data()
