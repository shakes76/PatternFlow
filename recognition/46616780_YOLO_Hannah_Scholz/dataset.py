# File containing the data loader for loading and preprocessing your data
from os import listdir
from PIL import Image

IMAGE_SIZE = 256


def load_images(image_path):

    file_names = listdir(image_path)
    print(file_names)

    for filename in file_names:
        if filename.endswith(".DS_Store"):
            continue

        with Image.open(image_path + filename) as img:
            width, height = img.size
            print(width)
            print(height)

            if width > height:
                img = img.resize((IMAGE_SIZE, round(height * (IMAGE_SIZE / width))))
            else:
                img = img.resize((round(height * (IMAGE_SIZE / width)), IMAGE_SIZE))


            bounding_box(img)

    # Change the shape of the images so all the images are the same size
    # And all the mask sizes too 256x256


# Steps to compute center:
# 1. Compute xMin, xMax, yMin, yMax
# 2. Average xMin and xMax, and yMin and yMax to compute the centre of bounding box
# 3. Divide the x average by the width of the image, and the y average by the height of the image
# Width and height computations are one-liners and are below
def bounding_box(image):
    print(image.size)


validation_path = "Datasets/Validation/validation_mask/"
# your images in an array
load_images(validation_path)
