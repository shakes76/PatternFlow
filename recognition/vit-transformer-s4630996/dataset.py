"""
Assumptions:
ADNI Data is from Blackboard
ADNI DATA has been unzipped
New folders for cropped images have been created
New folders for training and validation data copy from original have bee created
"""

########################################################################################################
###############################    Crop images from base ADNI dataset   ################################
########################################################################################################

from os import listdir
from PIL import Image


# get a list of files to crop
uncropped_files_train_AD = listdir(r"AD_NC\train\AD")
uncropped_files_train_NC = listdir(r"AD_NC\train\NC")
uncropped_files_test_AD = listdir(r"AD_NC\test\AD")
uncropped_files_test_NC = listdir(r"AD_NC\test\NC")

# define parameters for crop
left = 30
top = 10
right = 230
bottom = 210

# crop and save in new directory - train AD
for file in uncropped_files_train_AD:
    image = Image.open(r"AD_NC\train\AD\{}".format(file))
    cropped = image.crop((left, top, right, bottom))
    cropped.save(r"AD_NC_cropped\train\AD\{}".format(file))

# crop and save in new directory - train NC
for file in uncropped_files_train_NC:
    image = Image.open(r"AD_NC\train\NC\{}".format(file))
    cropped = image.crop((left, top, right, bottom))
    cropped.save(r"AD_NC_cropped\train\NC\{}".format(file))

# crop and save in new directory - test AD
for file in uncropped_files_test_AD:
    image = Image.open(r"AD_NC\test\AD\{}".format(file))
    cropped = image.crop((left, top, right, bottom))
    cropped.save(r"AD_NC_cropped\test\AD\{}".format(file))

# crop and save in new directory - test NC
for file in uncropped_files_test_NC:
    image = Image.open(r"AD_NC\test\NC\{}".format(file))
    cropped = image.crop((left, top, right, bottom))
    cropped.save(r"AD_NC_cropped\test\NC\{}".format(file))



########################################################################################################
###############################    Create Training and Validation Sets  ################################
########################################################################################################

from os import listdir
import random
import shutil

# import list of AD and NC file names from the training folder
train_AD_filenames = listdir(r"AD_NC_cropped\train\AD")
train_NC_filenames = listdir(r"AD_NC_cropped\train\NC")

# function to extract patient IDs from image filenames
def extract_patient_IDs(filenames):
    sep1 = "_"
    patients = []
    for filename in filenames:
        left, right = filename.split(sep1)
        ID = left
        if ID not in patients:
            patients.append(ID)
    return patients

# extract patient IDs from the training set for both classes
patients_AD = extract_patient_IDs(train_AD_filenames)
patients_NC = extract_patient_IDs(train_NC_filenames)

# subset the training set into training and validation sets 
train_split = 0.9
num_AD_train = round(len(patients_AD) * train_split)
num_NC_train = round(len(patients_NC) * train_split)
patients_AD_train = random.sample(patients_AD, num_AD_train)
patients_NC_train = random.sample(patients_NC, num_NC_train)

# each patient has 20 images or less - ensure no single patient is in both training and validation sets
# iterate over training images for AD class
for file in train_AD_filenames:
    # extract the patient id
    ID, _ = file.split("_")
    # if this id is in training subset, copy to that folder
    if ID in patients_AD_train:
        src = r"AD_NC_cropped\train\AD\{}".format(file)
        des = r"AD_NC_cropped\training\AD\{}".format(file)
        shutil.copy(src, des)
    # otherwise this patient is in the validation set so copy to validation folder  
    else:
        des = r"AD_NC_cropped\validation\AD\{}".format(file)
        shutil.copy(src, des)  
# repeat for NC class
for file in train_NC_filenames:
    ID, _ = file.split("_")
    print(ID)
    if ID in patients_NC_train:
        src = r"AD_NC_cropped\train\NC\{}".format(file)
        des = r"AD_NC_cropped\training\NC\{}".format(file)
        shutil.copy(src, des)
    else:
        des = r"AD_NC_cropped\validation\NC\{}".format(file)
        shutil.copy(src, des)  

