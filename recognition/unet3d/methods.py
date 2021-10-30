import os
import re
import nibabel as nib
# from matplotlib import pyplot as plt
from scipy import ndimage
import tensorflow as tf



def read_filename_from(directory):
    """"
    Read image names from target directory and sort by case no. then week no.
    """

    files = os.listdir(directory)
    files = [fname[:-12] for fname in files if fname.endswith('nii.gz')]
    files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    return files



def train_val_test_split(directory):
    """
    Classify images into dictionary by case number

    eg. {"case_004":['Case_004_Week0_LFOV.nii.gz', 'Case_004_Week1_LFOV.nii.gz', 'Case_004_Week2_LFOV.nii.gz'], 
    "case_002": [...]}
    """
    files = read_filename_from(directory)
    case_dict = {}
    for f in files:
        if f[:8] not in case_dict:
            case_dict[f[:8]] = [f]
        else:
            case_dict[f[:8]].append(f)
    train = []
    val = []
    test = []
    for key, cases in case_dict.items():
        num = len(cases)
        if num == 1:
            train.append(cases[0])
        if num == 2:
            train.append(cases[0])
            val.append(cases[1])
        if num >=3:
            train += cases[:-2]
            val.append(cases[-2])
            test.append(cases[-1])
    print("Numbers of cases for train, validation and test are: ", len(train), len(val), len(test), ", respectively")
    return train, val, test

def get_filepath_from(directory):
    """Get complete file paths for each set"""

    train, val, test = train_val_test_split(directory)

    return [os.path.join(directory, fname) for fname in train], [os.path.join(directory, fname) for fname in val], [os.path.join(directory, fname) for fname in test]
            

# def read_nii_file(path):

#     # read nii image as numpy data
#     return nib.load(path).get_fdata()

def read_nii_file(path):

    # read nii image as numpy data
    return nib.load(path).get_fdata()

    
def normalize(volume):
    """Interest area"""

    min, max = 0, 500
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume-min)/(max-min)
    return volume

def resize_volume(img):
    """Resize across z-axis"""
    desired_shape = (256, 256, 128)
    
    if img.shape == desired_shape:
        pass
    else: 
        # Get current depth
        current_width, current_height, current_depth = img.shape
        # Set the desired depth
        desired_width, desired_height, desired_depth = desired_shape
        # Compute depth factor
        depth_factor = desired_depth/current_depth
        width_factor = desired_width / current_width
        height_factor = desired_height / current_height
        # Resize across z-axis with bilinear interpolation
        img = ndimage.zoom(img.astype(int), (width_factor, height_factor, depth_factor), order=1)
    return img

def process_nii(path, norma = False):
    volume = read_nii_file(path)
    if norma:
        volume = normalize(volume)
    # volume = resize_volume(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume




