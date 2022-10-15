from tensorflow import keras
import os
import shutil

valid_directory = "valid-split"
train_directory = "train-split"
parent_dir = os.getcwd()

valid_path = os.path.join(parent_dir, valid_directory)
train_path = os.path.join(parent_dir, train_directory)

source_path = os.path.join(parent_dir, "train")

source_ad_path = os.path.join(source_path, "AD")
source_nc_path = os.path.join(source_path, "NC")

valid_ad = os.path.join(valid_path, "AD")
valid_nc = os.path.join(valid_path, "NC")

train_ad = os.path.join(train_path, "AD")
train_nc = os.path.join(train_path, "NC")

try:
    os.mkdir(valid_path)
    os.mkdir(valid_ad)
    os.mkdir(valid_nc)

    os.mkdir(train_path)
    os.mkdir(train_ad)
    os.mkdir(train_nc)

    ad_files = os.listdir(source_ad_path)

    split_flag = 0
    previous_id = 0
    for filename in ad_files:
        if filename.split("_")[0] == previous_id:
            if split_flag == 0:
                shutil.copy2(os.path.join(source_ad_path, filename), valid_ad)
            else:
                shutil.copy2(os.path.join(source_ad_path, filename), train_ad)
        else:
            if split_flag == 1:
                split_flag = 0
                shutil.copy2(os.path.join(source_ad_path, filename), valid_ad)
            else:
                split_flag = 1
                shutil.copy2(os.path.join(source_ad_path, filename), train_ad)

            previous_id = filename.split("_")[0]

    nc_files = os.listdir(source_nc_path)

    for filename in nc_files:
        if filename.split("_")[0] == previous_id:
            if split_flag == 0:
                shutil.copy2(os.path.join(source_nc_path, filename), valid_nc)
            else:
                shutil.copy2(os.path.join(source_nc_path, filename), train_nc)
        else:
            if split_flag == 1:
                split_flag = 0
                shutil.copy2(os.path.join(source_nc_path, filename), valid_nc)
            else:
                split_flag = 1
                shutil.copy2(os.path.join(source_nc_path, filename), train_nc)

            previous_id = filename.split("_")[0]

except OSError:
    pass

train = keras.utils.image_dataset_from_directory(
    directory='train-split/',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(256, 240))
valid = keras.utils.image_dataset_from_directory(
    directory='valid-split/',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(256, 240))

