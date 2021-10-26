import os
import re
import nibabel as nib
from matplotlib import pyplot as plt
import numpy as np
from scipy import ndimage
import tensorflow as tf



def read_filename_from(directory):
    """"Read image names from target directory and sort by case no. and week no."""

    files = os.listdir(directory)
    files = [fname for fname in files if fname.endswith('nii.gz')]
    files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    return files

def get_filepath_from(directory):
    """Get all file paths"""

    files = read_filename_from(directory)
    return [os.path.join(directory, fname) for fname in files]




def read_nii_file(path):
    return nib.load(path).get_fdata()

def normalize(volume):
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
        img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img

def process_nii(path, norma = False):
    volume = read_nii_file(path)
    if norma:
        volume = normalize(volume)
    volume = resize_volume(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume


# target_dir = "datasets/HipMRI_study_complete_release_v1/semantic_MRs_anon"
# label_dir = "datasets/HipMRI_study_complete_release_v1/semantic_labels_anon"

# mri_paths = get_filepath_from(target_dir)
# label_paths = get_filepath_from(label_dir)

# # volume = read_nii_file("datasets/HipMRI_study_complete_release_v1/semantic_MRs_anon/Case_019_Week1_LFOV.nii.gz")


# volume = process_nii(mri_paths[0])


# print(np.max(volume))
# print(np.min(volume))
# print(volume.shape)


# Build train and validation datasets

