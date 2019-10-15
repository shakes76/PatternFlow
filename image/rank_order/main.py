# Author: Askar Jaboldinov

from rank_order import rank_order
import os
from PIL import Image
import re

##Driver script
if __name__ == "__main__":
    user_input = input("Enter the path to the image file: ")
    # check if file exists
    assert os.path.exists(user_input), "No such file: "+str(user_input)
    # check if it is a jpeg file
    filename, file_extension = os.path.splitext(user_input)
    is_jpeg = re.search('.jpeg$', file_extension.lower())
    is_jpg = re.search('.jpg$', file_extension.lower())
    if is_jpeg or is_jpg:
        # get array
        converted = rank_order(user_input)
        # convert the array to image
        converted_img = Image.fromarray(rank_order(user_input)[0], 'RGB')
        converted_img.save("images/example_output.jpeg")
        print("converted file in the ./images/example_output.jpeg")
    else:
        print("only jpeg type files are supported! exiting module.")
