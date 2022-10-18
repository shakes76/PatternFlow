from sklearn.model_selection import train_test_split
import tensorflow as tf
from glob import glob
from sklearn.utils import shuffle
import os
#Contains the data loader for loading and preprocessing data

path = "C:/Users/danie/Downloads/ISIC DATA/"
im_height = 256
im_width = 256
batch_size = 4
learning_rate = 1e-4
num_epoch = 5

def load_data(path, split = 0.2):
    images = sorted(glob(os.path.join(path, "ISIC-2017_Training_Data", "*.jpg")))
    masks = sorted(glob(os.path.join(path, "ISIC-2017_Training_Part1_GroundTruth", "*.png")))

    test_size = int(len(images) * split)

    train_x, valid_x = train_test_split(images, test_size=test_size, random_state=42)
    train_y, valid_y = train_test_split(masks, test_size=test_size, random_state=42)

    train_x, test_x = train_test_split(train_x, test_size=test_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=test_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


if __name__ == "__main__":
    
    load_data(path=path)
    
    