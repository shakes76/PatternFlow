# File showing example usage of your trained model.
# Print out any results and / or provide visualisations where applicable
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image as im

IMAGE_SIZE = 640
EXAMPLE_IMAGE_PATH = "Datasets/Testing/testing_mask/ISIC_0012086_segmentation.png"


# Function that determines the bounding box of a given image (only used for the mask images in black and white)
def generate_bounding_box(image):
    height, width = image.size

    # Image colours are 0 = black, 255 = white
    # Lesion is in white
    pix = image.load()

    x_min = 640
    y_min = 640
    x_max = 0
    y_max = 0

    # Iterate through each pixel, the number of pixels is the size of image
    for y in range(IMAGE_SIZE):
        for x in range(IMAGE_SIZE):
            # Determine if each pixel is white and whether it is a max or min
            if pix[x, y] == 255:
                if x < x_min:
                    x_min = x
                elif x > x_max:
                    x_max = x
                if y < y_min:
                    y_min = y
                elif y > y_max:
                    y_max = y

    # Normalise
    x_avg_normalised = ((x_min + x_max) / 2) / IMAGE_SIZE
    y_avg_normalised = ((y_min + y_max) / 2) / IMAGE_SIZE

    # Bounding Box dimensions
    width_box = (x_max - x_min) / width
    height_box = (y_max - y_min) / height

    return x_avg_normalised, y_avg_normalised, width_box, height_box, x_min, x_max, y_min, y_max


# Function to show bounding box example on a photo
def bounding_box_figure(img):
    # Create figure and axes
    plt.imshow(img)
    ax = plt.gca()

    bounding_box_info = generate_bounding_box(img)

    # Bounding box width:
    bounding_width = bounding_box_info[2] * 640
    bounding_height = bounding_box_info[3] * 640
    x_min = bounding_box_info[4]
    y_min = bounding_box_info[6]

    # Create a Rectangle patch
    rect = Rectangle((x_min, y_min), bounding_width, bounding_height, linewidth=1, edgecolor='r', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)

    plt.show()


def main():
    with im.open(EXAMPLE_IMAGE_PATH) as img:
        bounding_box_figure(img)


if __name__ == '__main__':
    main()
