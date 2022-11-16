#this script splits the data to test and train with the ration 10 to 90 percent

import glob
import os
import numpy as np
import sys

current_dir = "./data/ISIS/images"
split_percent = 10;
train_list = open("data/ISIS/train.txt", "w")  
test_list = open("data/ISIS/val.txt", "w")  
counter = 1  
belong_to_test = round(100 / split_percent)  
for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.png")):  
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        if counter == belong_to_test:
                counter = 1
                test_list.write(current_dir + "/" + title + '.png' + "\n")
        else:
                train_list.write(current_dir + "/" + title + '.png' + "\n")
                counter = counter + 1
train_list.close()
test_list.close()