import os
import random
from PIL import Image
import numpy as np
from shutil import copyfile
import sys

IMAGE_DIR = 'ISIC2018_Task1-2_Training_Input_x2/photos/'
MASK_DIR = 'ISIC2018_Task1_Training_GroundTruth_x2/photos/'
NUM_FILES = 2594

# TODO: Python doc

def split_and_write_files(ids, new_dir, train_dir, valid_dir, file_ext, training_stop_id):
    index = 0
    ordered_file_names = os.listdir(new_dir)
    ordered_file_names.sort()
    for filename in ordered_file_names:
        if filename.endswith(file_ext):
            id = ids[index]
            src = new_dir + filename
            if id < training_stop_id:
                copyfile(src, train_dir + filename)
            else:
                copyfile(src, valid_dir + filename)
            index += 1

def training_validation_write(dir, valid_split):
    cwd = os.getcwd()
    img_train_dir = cwd + '/datasets/training/images/'
    img_valid_dir = cwd + '/datasets/validation/images/'
    mask_train_dir = cwd + '/datasets/training/masks/'
    mask_valid_dir = cwd + '/datasets/validation/masks/'
    # check if files already written
    if len(os.listdir(img_train_dir)) > 1:
        print('Dataset images already moved; skipping dataset copying')
        return img_train_dir, img_valid_dir, mask_train_dir, mask_valid_dir
    print('Copying ISICs training dataset files into local directory split into training and validation folders')
    ids = list(range(1, NUM_FILES + 1))
    random.seed(42)
    random.shuffle(ids)
    training_stop_id = round(NUM_FILES * (1 - valid_split))
    new_dir = dir + IMAGE_DIR
    split_and_write_files(ids, new_dir, img_train_dir, img_valid_dir, '.jpg', training_stop_id)
    new_dir = dir + MASK_DIR
    split_and_write_files(ids, new_dir, mask_train_dir, mask_valid_dir, '.png', training_stop_id)
    print('Copying complete')
    return img_train_dir, img_valid_dir, mask_train_dir, mask_valid_dir

# Steps to compute center:
# 1. Compute xMin, xMax, yMin, yMax
# 2. Average xMin and xMax, and yMin and yMax to compute the centre of bounding box
# 3. Divide the x average by the width of the image, and the y average by the height of the image
# Width and height computations are one-liners and are below
def generate_bounding_box(image):
    height, width = image.shape
    indices_of_mask_y = np.sort(np.argwhere(image>1)[:,0])
    indices_of_mask_x = np.sort(np.argwhere(image>1)[:,1])
    x_min = indices_of_mask_x[0]
    x_max = indices_of_mask_x[-1]
    y_min = height - indices_of_mask_y[-1]
    y_max = height - indices_of_mask_y[0] # since top left is 0

    norm_x_avg = ((x_min + x_max) / 2) / width
    norm_y_avg = ((y_min + y_max) / 2) / height
    box_width = (x_max - x_min) / width
    box_height = (y_max - y_min) / height

    return norm_x_avg, norm_y_avg, box_width, box_height

def generate_mask_datum(img_dir, mask_dir):
    files_created = 0
    for filename in os.listdir(mask_dir):
        if filename.endswith('.png'): # masks are .png for ISICs
            image = np.array(Image.open(mask_dir + filename))
            bound_box_info = generate_bounding_box(image)
            image_filename = filename[0:12]
            file = open(img_dir + image_filename + '.txt', 'w')
            file.write(f'0 {bound_box_info[0]} {bound_box_info[1]} {bound_box_info[2]} {bound_box_info[3]}')
            files_created += 1
    return files_created

def mask_info_cleanup(img_dir):
    deleted_files = 0
    files = os.listdir(img_dir)
    for filename in files:
        if filename.endswith('.txt'):
            os.remove(img_dir + filename)
            deleted_files += 1
    return deleted_files

# File format must be each row being: class x_center y_center width height,
# normalised to image dimensions with the top left corner being (0,0) and the bottom right corner as (1,1)
def generate_mask_data(img_train_dir, img_valid_dir, mask_train_dir, mask_valid_dir):
    print('Deleting previously generated mask data') # cleaned instead of skipping as old way may be wrong
    deleted_files = mask_info_cleanup(img_train_dir) + mask_info_cleanup(img_valid_dir)
    print(f'Finished deleting {deleted_files} files')
    print('Generating mask data for YOLOv5')
    files_created = generate_mask_datum(img_train_dir, mask_train_dir) + generate_mask_datum(img_valid_dir, mask_valid_dir)
    print(f'Finished generating {files_created} mask data files; exiting')

def main(args):
    img_train_dir, img_valid_dir, mask_train_dir, mask_valid_dir = training_validation_write(args[0], args[1])
    generate_mask_data(img_train_dir, img_valid_dir, mask_train_dir, mask_valid_dir)

if __name__ == '__main__':
    print('Program takes inputs of the form: current_dataset_directory, validation_data_split')
    if len(sys.argv) > 2:
        main(sys.argv[1:])
    else:
        print('Using default parameters')
        dir = os.path.expanduser('~/Datasets/ISIC2018_Task1-2_Training_Data/')
        valid_split = 0.2
        main([dir, valid_split])