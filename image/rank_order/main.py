# Author: Askar Jaboldinov

import rank_order
import sys
import os

##Driver script
if __name__ == "__main__":
    user_input = raw_input("Enter the path to the image file: ")

    assert os.path.exists(user_input), "No such file: "+str(user_input)
    converted = rank_order(user_input)
