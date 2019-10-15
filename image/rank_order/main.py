# Author: Askar Jaboldinov

from rank_order import rank_order
import sys
import os
from PIL import Image

##Driver script
if __name__ == "__main__":
    user_input = input("Enter the path to the image file: ")

    assert os.path.exists(user_input), "No such file: "+str(user_input)
    converted = rank_order(user_input)
    converted_img = Image.fromarray(rank_order(user_input)[0], 'RGB')
    converted_img.show()
