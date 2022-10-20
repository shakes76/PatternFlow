
import Augmentor
import os
import shutil
from os import listdir
from PIL import Image
import random

class ADNI_dataset_utility:
    """
    # make new folder structure
    # Extract images from ADNI dataset
    # if crop is true then crop
    # if augment is true then augment
    # make a validation set

    """
    def __init__(self):
        self.crop = True
        self.augment = True
        self.num_samples = 10000
        self.folders_top = ["AD_NC_new"]
        self.folders_middle = ["test", "training", "validation"]
        self.folders_bottom = ["AD", "NC"]
        self.validation_split = 0.1
        self.steps = ['create_directories', "move_and_split_images", "crop_images", "augment_images"]
        self.parent_directory = os.getcwd()

    def extract_patient_IDs(self, filenames):
        sep1 = "_"
        patient_ids = []
        for filename in filenames:
            left, right = filename.split(sep1)
            ID = left
            if ID not in patient_ids:
                patient_ids.append(ID)

        return patient_ids

    def create_directories(self):
        # make new folder structure
        # create folder structure
        # parent_directory = os.getcwd()

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
        
        # create folder for checkpoint
        path_checkpoint = os.path.join(self.parent_directory, "checkpoint")
        os.mkdir(path_checkpoint)
            
            
    def move_and_split_images(self):
        """ assumes new directory structure in place"""
        
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

        # extract images from original train folder and move to either training or train in new folder structure

        for file in train_AD_filenames:
            ID, _ = file.split("_")
            src = r"AD_NC\train\AD\{}".format(file)
            if ID in patients_AD_train:
                des = r"{}\training\AD\{}".format(self.folders_top[0], file)
            else:
                des = r"{}\validation\AD\{}".format(self.folders_top[0], file)
            shutil.copy(src, des)  

        for file in train_NC_filenames:
            ID, _ = file.split("_")
            src = r"AD_NC\train\NC\{}".format(file)
            if ID in patients_NC_train:
                des = r"{}\training\NC\{}".format(self.folders_top[0], file)
            else:
                des = r"{}\validation\NC\{}".format(self.folders_top[0], file)
            shutil.copy(src, des)  

        # copy images from test folder to new test folder
        test_NC_filenames = os.listdir(r"AD_NC\test\NC")
        test_AD_filenames = os.listdir(r"AD_NC\test\AD")

        for file in test_NC_filenames:
            src = r"AD_NC\test\NC\{}".format(file)
            des = r"{}\test\NC\{}".format(self.folders_top[0], file)
            shutil.copy(src, des)

        for file in test_AD_filenames:
            src = r"AD_NC\test\AD\{}".format(file)
            des = r"{}\test\AD\{}".format(self.folders_top[0], file)
            shutil.copy(src, des)
 
    def create_folder_structure(self, folders_top):
        # create folder structure
        parent_directory = os.getcwd()
        folders_top = folders_top
        folders_middle = ["test", "training", "validation"]
        folders_bottom = ["AD", "NC"]

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

        # create folder structure
        parent_directory = os.getcwd()
        folders_top = ["AD_NC_square"]
        folders_middle = ["test", "training", "validation"]
        folders_bottom = ["AD", "NC"]

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
        uncropped_files_training_AD = os.listdir(r"AD_NC_new\training\AD")
        uncropped_files_training_NC = os.listdir(r"AD_NC_new\training\NC")
        uncropped_files_validation_AD = os.listdir(r"AD_NC_new\validation\AD")
        uncropped_files_validation_NC = os.listdir(r"AD_NC_new\validation\NC")
        uncropped_files_test_AD = os.listdir(r"AD_NC_new\test\AD")
        uncropped_files_test_NC = os.listdir(r"AD_NC_new\test\NC")

        left = 8
        top = 0
        right = 248
        bottom = 240

        for file in uncropped_files_training_AD:
            image = Image.open(r"AD_NC_new\training\AD\{}".format(file))
            cropped = image.crop((left, top, right, bottom))
            cropped.save(r"AD_NC_square\training\AD\{}".format(file))

        for file in uncropped_files_training_NC:
            image = Image.open(r"AD_NC_new\training\NC\{}".format(file))
            cropped = image.crop((left, top, right, bottom))
            cropped.save(r"AD_NC_square\training\NC\{}".format(file))

        for file in uncropped_files_validation_AD:
            image = Image.open(r"AD_NC_new\validation\AD\{}".format(file))
            cropped = image.crop((left, top, right, bottom))
            cropped.save(r"AD_NC_square\validation\AD\{}".format(file))

        for file in uncropped_files_validation_NC:
            image = Image.open(r"AD_NC_new\validation\NC\{}".format(file))
            cropped = image.crop((left, top, right, bottom))
            cropped.save(r"AD_NC_square\validation\NC\{}".format(file))
            
        for file in uncropped_files_test_AD:
            image = Image.open(r"AD_NC_new\test\AD\{}".format(file))
            cropped = image.crop((left, top, right, bottom))
            cropped.save(r"AD_NC_square\test\AD\{}".format(file))

        for file in uncropped_files_test_NC:
            image = Image.open(r"AD_NC_new\test\NC\{}".format(file))
            cropped = image.crop((left, top, right, bottom))
            cropped.save(r"AD_NC_square\test\NC\{}".format(file))

    def add_augmented_images(self):

        # create folder structure
        self.create_folder_structure(["AD_NC_aug"])

        # copy existing files into new folder structure
        training_NC_filenames = os.listdir(r"AD_NC_square\training\NC")
        training_AD_filenames = os.listdir(r"AD_NC_square\training\AD")

        for file in training_NC_filenames:
            src = r"AD_NC_square\training\NC\{}".format(file)
            des = r"AD_NC_aug\training\NC\{}".format(file)
            shutil.copy(src, des)

        for file in training_AD_filenames:
            src = r"AD_NC_square\training\AD\{}".format(file)
            des = r"AD_NC_aug\training\AD\{}".format(file)
            shutil.copy(src, des)

        # define source and destination paths
        src_path = r"C:\Users\lovet\Documents\COMP3710\Report\AD_NC_aug\training\AD"
        dest_path = r"C:\Users\lovet\Documents\COMP3710\Report\AD_NC_aug\training\AD"

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
        src_path = r"C:\Users\lovet\Documents\COMP3710\Report\AD_NC_aug\training\NC"
        dest_path = r"C:\Users\lovet\Documents\COMP3710\Report\AD_NC_aug\training\NC"

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