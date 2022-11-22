import os
from PIL import Image
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Constants
IMAGE_DIM = 640

def resize_images(folder_path, dim, verbose = False):
    # Loop through the designated folder and look for images to resize
    for dirname, _, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirname, filename)
            if '.jpg' in file_path or 'segmentation.png' in file_path:
                # Resize and save images
                resized_img = (Image.open(file_path)).resize((dim, dim))
                if verbose:
                    print(file_path[:-4] + f'_resized_{dim}' + file_path[-4:])
                resized_img.save(file_path[:-4] + f'_resized_{dim}' + file_path[-4:])

def get_bounding_box(image_path):
    image = np.array(Image.open(image_path))
    
    min_x = image.shape[0]
    min_y = image.shape[1]
    max_x = 0
    max_y = 0

    # Iterate through the image
    for row in range(image.shape[1]):
        for col in range(image.shape[0]):
            if image[row][col] != 0:
                min_x = min(min_x, col)
                min_y = min(min_y, row)
                max_x = max(max_x, col)
                max_y = max(max_y, row)

    # Find width, height and center coords
    w = max_x - min_x
    h = max_y - min_y

    center_x = min_x + (w/2)
    center_y = min_y + (h/2)
    return np.append(np.array([center_x, center_y, w, h])/image.shape[0], [min_x, max_x, min_y, max_y])

def create_yolo_label(seg_path, csv_path):
    # Loop through the designated folder and look for images to resize
    for dirname, _, filenames in os.walk(seg_path):
        for filename in filenames:
            file_path = os.path.join(dirname, filename)
            if 'segmentation' in file_path:
                box = get_bounding_box(file_path)
                img_id = filename.replace('_segmentation_resized_640.png', '')
                print(img_id)
                # Open Truth CSV file
                truth = csv.reader(open(csv_path))

                # Write entry to .txt file for positive cases of melanoma
                for entry in truth:
                    if entry[0] == img_id:
                        print(seg_path + '\Test_data\labels\\' + img_id + '_resized_640' + '.txt')
                        label = open(img_id + '_resized_640' + '.txt', 'w')
                        label.write(f'{entry[1]} {box[0]} {box[1]} {box[2]} {box[3]}')


def plot_bounding_box(img_path, seg_path):
    # Get Bounding box info and open image based on path
    box_info = get_bounding_box(seg_path)
    img = Image.open(img_path)

    # Create figure and axes
    plt.imshow(img)
    ax = plt.gca()

    # Create a Rectangular Patch and Add to Plot
    rect = Rectangle((box_info[4], box_info[6]), box_info[2] * img.shape[0], box_info[3] * img.shape[1], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    plt.show()
        

def main():
    resize_images("D:\GitHub\COMP3710 DEMO 3\ISIC", IMAGE_DIM, verbose=True)
    create_yolo_label("D:\Dataset\ISIC_640\Train", "D:\Dataset\ISIC_640\Train\Train_Truth_gold.csv")
    # Used to check if bounding boxes are places correctly
    plot_bounding_box("D:\Dataset\ISIC_640\Train\Train_data\images\ISIC_0000000_resized_640.jpg", "D:\Dataset\ISIC_640\Train\Train_Truth_PNG\ISIC-2017_Training_Part1_GroundTruth\ISIC_0000000_segmentation_resized_640.png")

if __name__ == "__main__":
    main()