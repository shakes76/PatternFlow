"""
This script writes to "train.txt" and "test.txt", all the train and test image paths respectively.

@author Lalith Veerabhadrappa Badiger
@email l.badiger@uqconnect.edu.au
"""

import os

def generate_train():
    """
    Writes to train.txt, all the train image paths.
    """
    image_files = []
    os.chdir(os.path.join("data", "obj"))
    for filename in os.listdir(os.getcwd()):
        if filename.endswith(".jpg"):
            image_files.append("data/obj/" + filename)
    os.chdir("..")
    with open("train.txt", "w") as outfile:
        for image in image_files:
            outfile.write(image)
            outfile.write("\n")
        outfile.close()
    os.chdir("..")


def generate_test():
    """
    Writes to test.txt, all the test image paths.
    """
    image_files = []
    os.chdir(os.path.join("data", "valid"))
    for filename in os.listdir(os.getcwd()):
        if filename.endswith(".jpg"):
            image_files.append("data/valid/" + filename)
    os.chdir("..")
    with open("test.txt", "w") as outfile:
        for image in image_files:
            outfile.write(image)
            outfile.write("\n")
        outfile.close()
    os.chdir("..")

generate_train()
generate_test()