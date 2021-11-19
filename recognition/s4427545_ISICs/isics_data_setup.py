import os
import random
from PIL import Image as im
import numpy as np
import sys

IMAGE_DIR = 'ISIC2018_Task1-2_Training_Input_x2/photos/'
MASK_DIR = 'ISIC2018_Task1_Training_GroundTruth_x2/photos/'
NUM_FILES = 2594
IMG_SIZE = 640
AUGMENTATIONS = 2

# Should make 6 times the data by doing all rotations and a flip from the base, and even cropping...
def split_and_write_files(ids, old_img_dir, img_dirs, mask_dir, file_ext, training_stop_id, valid_stop_id):
    train_dirs = [img_dirs[0], mask_dir, img_dirs[0].replace('images', 'labels')]
    index = 0
    ordered_file_names = os.listdir(old_img_dir)
    ordered_file_names.sort()
    files_written = 0
    for filename in ordered_file_names:
        if filename.endswith(file_ext):
            id = ids[index]
            src = old_img_dir + filename
            with im.open(src) as image:
                width, height = image.size
                # resize with max axis being 640 and keep aspect ratio
                if width > height:
                    image = image.resize((IMG_SIZE, round(height * (IMG_SIZE/width))))
                else:
                    image = image.resize((round(width * (IMG_SIZE/height)), IMG_SIZE))
                if id < training_stop_id:
                    files_written += save_img_with_augmentations(image, train_dirs[0], train_dirs[1] + filename[0:12] + '_segmentation.png', train_dirs[2], filename)
                elif id < valid_stop_id:
                    image.save(img_dirs[1] + filename)
                    files_written += 1
                else:
                    image.save(img_dirs[2] + filename)
                    files_written += 1
            index += 1
    print(f'{files_written} files written')

def save_img_with_augmentations(image : im.Image, new_image_dir, mask_dir, new_label_dir, filename):
    # Have 6 times more training data. But, be careful not to contaminate!!!
    name = filename.replace('.jpg', '')
    images = {}
    images['base'] = image
    images['rot90'] = image.rotate(90,expand=True)
    images['rot180'] = image.rotate(180)
    images['rot270'] = image.rotate(270,expand=True)
    images['horiz_flip'] = image.transpose(method=im.FLIP_LEFT_RIGHT)
    images['vertic_flip'] = image.transpose(method=im.FLIP_TOP_BOTTOM)
    masks = {}
    with im.open(mask_dir) as mask_image:
        masks['base'] = mask_image
        masks['rot90'] = mask_image.rotate(90,expand=True)
        masks['rot180'] = mask_image.rotate(180)
        masks['rot270'] = mask_image.rotate(270,expand=True)
        masks['horiz_flip'] = mask_image.transpose(method=im.FLIP_LEFT_RIGHT)
        masks['vertic_flip'] = mask_image.transpose(method=im.FLIP_TOP_BOTTOM)
        keys = list(images.keys())
        random.shuffle(keys)
        keys = keys[0:AUGMENTATIONS]
        for key in keys:
            new_image = images[key]
            new_mask = masks[key]
            image_arr = np.array(new_mask)
            bound_box_info = generate_bounding_box(image_arr)
            new_name = name + '_' + key
            file = open(new_label_dir + new_name + '.txt', 'w')
            file.write(f'0 {bound_box_info[0]} {bound_box_info[1]} {bound_box_info[2]} {bound_box_info[3]}')
            new_image.save(new_image_dir + new_name + '.jpg')
    return AUGMENTATIONS * 2


def training_validation_test_write(dir, valid_split, test_split):
    print('Copying ISICs training dataset files into local directory split into training, validation, and test folders')
    cwd = os.getcwd()
    img_train_dir = cwd + '/datasets/ISIC/training/images/'
    img_valid_dir = cwd + '/datasets/ISIC/validation/images/'
    img_test_dir = cwd + '/datasets/ISIC/test/images/'
    labels_train_dir = cwd + '/datasets/ISIC/training/labels/'
    labels_valid_dir = cwd + '/datasets/ISIC/validation/labels/'
    labels_test_dir = cwd + '/datasets/ISIC/test/labels/'

    ids = list(range(1, NUM_FILES + 1))
    random.seed(42)
    random.shuffle(ids)

    print('Deleting previously generated data') # cleaned instead of skipping as old way may be wrong
    deleted_files = cleanup(img_train_dir) + cleanup(img_valid_dir) + cleanup(img_test_dir)
    deleted_files += cleanup(labels_train_dir) + cleanup(labels_valid_dir) + cleanup(labels_test_dir)
    print(f'Finished deleting {deleted_files} files')

    valid_stop_id = round(NUM_FILES * (1 - test_split))
    training_stop_id = round(valid_stop_id * (1 - valid_split))
    old_img_dir = dir + IMAGE_DIR
    old_mask_dir = dir + MASK_DIR
    split_and_write_files(ids, old_img_dir, [img_train_dir, img_valid_dir, img_test_dir], old_mask_dir, '.jpg', training_stop_id, valid_stop_id)
    print('Copying complete')

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

def cleanup(img_dir):
    deleted_files = 0
    files = os.listdir(img_dir)
    for filename in files:
        os.remove(img_dir + filename)
        deleted_files += 1
    return deleted_files

def main(args):
    training_validation_test_write(args[0], args[1], args[2])

if __name__ == '__main__':
    print('Program takes inputs of the form: current_dataset_directory, validation_data_split')
    if len(sys.argv) > 2:
        main(sys.argv[1:])
    else:
        print('Using default parameters')
        dir = os.path.expanduser('~/Datasets/ISIC2018_Task1-2_Training_Data/')
        valid_split = 0.2
        test_split = 0.2
        main([dir, valid_split, test_split])