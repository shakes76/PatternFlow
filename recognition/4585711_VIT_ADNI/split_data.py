import os
import random

from utils import Params

def move_files(directory, valid_dir):
    files = dict()
    for file in os.listdir(directory):
        key = int(file.split("_")[0])

        if key in files:
            files[key].add(file)
        else:
            files[key] = set()
    
    file_count = 0
    keys = list(files.keys())
    while file_count < 3000:
        key = random.choice(keys)
        keys.remove(key)
        file_set = files[key]

        for file in file_set:
            os.replace(directory + file, valid_dir + file)
        
        file_count += len(file_set)

if __name__ == "__main__":
    p = Params()

    data_dir = p.data_dir() + "AD_NC/"

    os.makedirs(data_dir + "valid/AD")
    os.makedirs(data_dir + "valid/NC")

    test_AD_dir = data_dir + "test/AD/"
    test_NC_dir = data_dir + "test/NC/"

    valid_AD_dir = data_dir + "valid/AD/"
    valid_NC_dir = data_dir + "valid/NC/"

    move_files(test_AD_dir, valid_AD_dir)
    move_files(test_NC_dir, valid_NC_dir)