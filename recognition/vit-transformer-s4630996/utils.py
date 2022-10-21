import Augmentor
import os
import shutil
from os import listdir
from PIL import Image
import random
from config import *

random.seed(20)

class ADNI_dataset_utility:
    """
        Three key methods:
            1) create_directories() to make new folder structure
            2) move_and_split_images() to create the validation split
            3) crop_and_save_images() to crop images to square 240 x 240
            4) add_augmented_images() to add augmented images to training set
    """
    def __init__(self):
        self.crop = True
        self.augment = True
        self.num_samples = NUM_AUG_SAMPLES
        self.folders_top = ["AD_NC_split"]
        self.folders_middle = ["test", "training", "validation"]
        self.folders_bottom = ["AD", "NC"]
        self.validation_split = VALIDATION_SPLIT
        self.steps = ['create_directories', "move_and_split_images", "crop_images", "augment_images"]
        self.parent_directory = os.getcwd()

    def extract_patient_IDs(self, filenames):
        """ extract a set of patient IDs from a list of ADNI filenames"""
        # patient id is separated by "_"
        sep1 = "_"
        patient_ids = []
        # iterate over filenames
        for filename in filenames:
            # grabe the ID
            ID, _ = filename.split(sep1)
            # create a set of patient IDs
            if ID not in patient_ids:
                patient_ids.append(ID)

        return patient_ids

    def create_directories(self):
        """ creates ADNI image folder stucture based on utility class properties plus
        folders for model checkpoint and plots"""

        # create folders for image data
        for folder_top in self.folders_top:
            path_top = os.path.join(self.parent_directory, folder_top)
            os.mkdir(path_top)
            for folder_middle in self.folders_middle:
                path_middle = os.path.join(path_top, folder_middle)
                os.mkdir(path_middle)
                for folder_bottom in self.folders_bottom:
                    path_bottom = os.path.join(path_middle, folder_bottom)
                    os.mkdir(path_bottom)
        
        # create folder to save model weights (checkpoint)
        path_checkpoint = os.path.join(self.parent_directory, "checkpoint")
        os.mkdir(path_checkpoint)

        # create folder to save plots
        path_plots = os.path.join(self.parent_directory, "plots")
        os.mkdir(path_plots)

            
    def move_and_split_images(self):
        """ creates a validation split from the training set"""
        
        top_folder = "AD_NC_split"

        # import list of AD and NC file names from the train folder
        train_AD_filenames = os.listdir(r"AD_NC\train\AD")
        train_NC_filenames = os.listdir(r"AD_NC\train\NC")

        # extract patient ids for each directory
        patient_ids_AD = self.extract_patient_IDs(train_AD_filenames)
        patient_ids_NC = self.extract_patient_IDs(train_NC_filenames)
        
        # identify ids for the training set
        train_split = 1 - self.validation_split
        num_AD_train = round(len(patient_ids_AD) * train_split)
        num_NC_train = round(len(patient_ids_NC) * train_split)

        # take a random sample for the training set
        patients_AD_train = random.sample(patient_ids_AD, num_AD_train)
        patients_NC_train = random.sample(patient_ids_NC, num_NC_train)

        # extract images from original train folder 
        for file in train_AD_filenames:
            ID, _ = file.split("_")
            # move to either training or validation folder
            src = r"AD_NC\train\AD\{}".format(file)
            if ID in patients_AD_train:
                des = r"{}\training\AD\{}".format(top_folder, file)
            else:
                des = r"{}\validation\AD\{}".format(top_folder, file)
            shutil.copy(src, des)  
        
        # repeat for NC folder
        for file in train_NC_filenames:
            ID, _ = file.split("_")
            src = r"AD_NC\train\NC\{}".format(file)
            if ID in patients_NC_train:
                des = r"{}\training\NC\{}".format(top_folder, file)
            else:
                des = r"{}\validation\NC\{}".format(top_folder, file)
            shutil.copy(src, des)  

        # copy images from test folder to new test folder
        test_NC_filenames = os.listdir(r"AD_NC\test\NC")
        test_AD_filenames = os.listdir(r"AD_NC\test\AD")

        # straightfoward copy of NC images from test to test
        for file in test_NC_filenames:
            src = r"AD_NC\test\NC\{}".format(file)
            des = r"{}\test\NC\{}".format(top_folder, file)
            shutil.copy(src, des)

        # repeat for AD
        for file in test_AD_filenames:
            src = r"AD_NC\test\AD\{}".format(file)
            des = r"{}\test\AD\{}".format(top_folder, file)
            shutil.copy(src, des)
 
    def create_folder_structure(self, folders_top):
        """ same as create directories above except can define a different top folder"""
        parent_directory = os.getcwd()
        folders_top = folders_top
        folders_middle = ["test", "training", "validation"]
        folders_bottom = ["AD", "NC"]

        # iterate over folders to create directory structure
        for folder_top in folders_top:
            path_top = os.path.join(parent_directory, folder_top)
            os.mkdir(path_top)
            for folder_middle in folders_middle:
                path_middle = os.path.join(path_top, folder_middle)
                os.mkdir(path_middle)
                for folder_bottom in folders_bottom:
                    path_bottom = os.path.join(path_middle, folder_bottom)
                    os.mkdir(path_bottom)


    def crop_and_save_images(self):
        """crop ADNI images to square 240 x 240 - extracts images from AD_NC top folder
        and creates new folder structure under AD_NC_square"""

        # identify folder names
        parent_directory = os.getcwd()
        folders_top = ["AD_NC_square"]
        folders_middle = ["test", "training", "validation"]
        folders_bottom = ["AD", "NC"]
        
        # create folder structure
        for folder_top in folders_top:
            path_top = os.path.join(parent_directory, folder_top)
            os.mkdir(path_top)
            for folder_middle in folders_middle:
                path_middle = os.path.join(path_top, folder_middle)
                os.mkdir(path_middle)
                for folder_bottom in folders_bottom:
                    path_bottom = os.path.join(path_middle, folder_bottom)
                    os.mkdir(path_bottom)


        # get a list of files to crop
        uncropped_files_training_AD = os.listdir(r"AD_NC_split\training\AD")
        uncropped_files_training_NC = os.listdir(r"AD_NC_split\training\NC")
        uncropped_files_validation_AD = os.listdir(r"AD_NC_split\validation\AD")
        uncropped_files_validation_NC = os.listdir(r"AD_NC_split\validation\NC")
        uncropped_files_test_AD = os.listdir(r"AD_NC_split\test\AD")
        uncropped_files_test_NC = os.listdir(r"AD_NC_split\test\NC")

        # crop specs to create 240 x 240 from 256 x 240
        left = 8
        top = 0
        right = 248
        bottom = 240

        # crop training AD images
        for file in uncropped_files_training_AD:
            image = Image.open(r"AD_NC_split\training\AD\{}".format(file))
            cropped = image.crop((left, top, right, bottom))
            cropped.save(r"AD_NC_square\training\AD\{}".format(file))

        # crop training NC images
        for file in uncropped_files_training_NC:
            image = Image.open(r"AD_NC_split\training\NC\{}".format(file))
            cropped = image.crop((left, top, right, bottom))
            cropped.save(r"AD_NC_square\training\NC\{}".format(file))

        # crop validation AD images
        for file in uncropped_files_validation_AD:
            image = Image.open(r"AD_NC_split\validation\AD\{}".format(file))
            cropped = image.crop((left, top, right, bottom))
            cropped.save(r"AD_NC_square\validation\AD\{}".format(file))

        # crop training NC images
        for file in uncropped_files_validation_NC:
            image = Image.open(r"AD_NC_split\validation\NC\{}".format(file))
            cropped = image.crop((left, top, right, bottom))
            cropped.save(r"AD_NC_square\validation\NC\{}".format(file))
    
        # crop test AD images
        for file in uncropped_files_test_AD:
            image = Image.open(r"AD_NC_split\test\AD\{}".format(file))
            cropped = image.crop((left, top, right, bottom))
            cropped.save(r"AD_NC_square\test\AD\{}".format(file))

        # crop test NC images
        for file in uncropped_files_test_NC:
            image = Image.open(r"AD_NC_split\test\NC\{}".format(file))
            cropped = image.crop((left, top, right, bottom))
            cropped.save(r"AD_NC_square\test\NC\{}".format(file))

    def add_augmented_images(self):
        """ add augmnted images to training images - uses Augmentor module"""

        parent_directory = os.getcwd()

        # define source and destination paths
        src_path = r"AD_NC_square\training\AD"
        dest_path = r"AD_NC_square\training\AD"
        src_path = os.path.join(parent_directory, src_path)
        dest_path = os.path.join(parent_directory, dest_path)

        # instantiate Augmentor pipeline and add augmentations
        p = Augmentor.Pipeline(source_directory=src_path, output_directory=dest_path)
        p.skew(probability=0.3, magnitude=0.1)
        p.shear(probability=0.3, max_shear_left=5, max_shear_right=5)
        p.random_erasing(probability=0.3, rectangle_area=0.1)
        p.histogram_equalisation(probability=0.2)
        p.flip_left_right(probability=0.2)
        p.crop_random(probability=0.3, percentage_area=0.95, randomise_percentage_area=False)
        p.random_brightness(probability=0.4, min_factor=0.3, max_factor=1.2)  # 0 black, 1 original
        p.rotate_without_crop(probability=0.4, max_left_rotation=5, max_right_rotation=5)
        p.sample(self.num_samples)

        # repeat for NC data
        src_path = r"AD_NC_square\training\NC"
        dest_path = r"AD_NC_square\training\NC"
        src_path = os.path.join(parent_directory, src_path)
        dest_path = os.path.join(parent_directory, dest_path)

        # instantiate Augmentor pipeline and add augmentations
        p = Augmentor.Pipeline(source_directory=src_path, output_directory=dest_path)
        p.skew(probability=0.3, magnitude=0.1)
        p.shear(probability=0.3, max_shear_left=5, max_shear_right=5)
        p.random_erasing(probability=0.3, rectangle_area=0.1)
        p.histogram_equalisation(probability=0.2)
        p.flip_left_right(probability=0.2)
        p.crop_random(probability=0.3, percentage_area=0.95, randomise_percentage_area=False)
        p.random_brightness(probability=0.4, min_factor=0.3, max_factor=1.2)  # 0 black, 1 original
        p.rotate_without_crop(probability=0.4, max_left_rotation=5, max_right_rotation=5)
        p.sample(self.num_samples)


def main():

    # extract a new dataset utility object
    new_utility = ADNI_dataset_utility()
    
    # create directories
    new_utility.create_directories()

    # move and split images
    new_utility.move_and_split_images()

    # crop to square
    new_utility.crop_and_save_images()

    # augment images into new folder structures
    new_utility.add_augmented_images()


if __name__ == "__main__":
    main()