from pyimgaug3d.augmentation import GridWarp, Flip
from pyimgaug3d.augmenters import ImageSegmentationAugmenter
import nibabel as nib
import numpy as np
import fnmatch
import os
import tensorflow as tf


# Credit: Taken from demo.py in pyimgaug3d
def save_as_nifti(data, folder, name, affine=np.eye(4)):
    img = nib.Nifti1Image(data, affine)
    if not os.path.exists(folder):
        os.mkdir(folder)
    nib.save(img, os.path.join(folder, name))
    pass


# Pre-process and Augment the CSIRO 3D Male Pelvic dataset
# Original data is expected in two sub-folders, but is outputted to a single directory
# Assumption: Both X and Y Folders contain ONLY training data and independently sorting both folders
# alphabetically will result in the i-th file in both folders being an X,Y pair (for all i)
def data_preprocess_augment(orig_data_dirpath, orig_data_x_subdirname, orig_data_y_subdirname, output_data_dirpath,
                            ds_factor=1, aug_count=0, verbose=False):
    expected_img_size = (256, 256, 128)
    # Set up paths
    orig_data_x_dirpath = orig_data_dirpath + '/' + orig_data_x_subdirname
    orig_data_y_dirpath = orig_data_dirpath + '/' + orig_data_y_subdirname
    # Fetch .nii.gz files
    x_filenames = np.array(fnmatch.filter(sorted(os.listdir(orig_data_x_dirpath)), '*.nii.gz'))
    y_filenames = np.array(fnmatch.filter(sorted(os.listdir(orig_data_y_dirpath)), '*.nii.gz'))
    # Filter out any marked as AUG (Augmented), returning only the originals
    x_filenames = x_filenames[np.array([not fnmatch.fnmatch(filename, '*AUG*') for filename in x_filenames])]
    y_filenames = y_filenames[np.array([not fnmatch.fnmatch(filename, '*AUG*') for filename in y_filenames])]
    # Combine into a 2D Array
    data_filenames = np.vstack((x_filenames, y_filenames)).T
    num_files = len(data_filenames)
    # Make the output directory if needed
    if not os.path.exists(output_data_dirpath):
        os.mkdir(output_data_dirpath)
    # Lambda function to merge all classes except 5 (Prostate)
    simplify_labels = np.vectorize(lambda x: float(x == 5))
    # Process each image
    for idx, datum in enumerate(data_filenames):
        print('Processing observation #{} of {}: {}'.format(idx + 1, num_files, datum))
        # Load X Y file pair into memory
        curr_x = nib.load(orig_data_x_dirpath + '/' + datum[0]).get_fdata()
        curr_y = nib.load(orig_data_y_dirpath + '/' + datum[1]).get_fdata()
        print("unique vals in y:", np.unique(curr_y))
        ################################################################################################################
        # DATA CLEANSING AND DOWN-SAMPLING #############################################################################
        ################################################################################################################
        # Check X Y shapes match
        if curr_x.shape != curr_y.shape:
            print("X,Y have different shapes. Skipping...") if verbose else None
            continue
        # Check dimensions match expectation
        if len(curr_x.shape) != len(expected_img_size):
            print("X,Y have incompatible dimensions. Skipping...") if verbose else None
            continue
        # Find unexpected shapes
        size_diffs = np.subtract(curr_x.shape, expected_img_size)
        # Skip if image is smaller than expected
        if np.min(size_diffs) < 0:
            print("Image is smaller than expected. Skipping...") if verbose else None
            continue
        # Trim if image is larger than expected
        if np.max(size_diffs) > 0:
            print("Image is larger than expected. Trimming edges...") if verbose else None
            # Calculate how much to trim off the edges to match desired shape
            # If deviation is odd, trims one more pixel from the end to compensate
            trims = [[int(np.floor(i / 2)), int(np.ceil(i / 2))] for i in size_diffs]
            shp = curr_x.shape
            curr_x = curr_x[
                     trims[0][0]:shp[0] - trims[0][1],
                     trims[1][0]:shp[1] - trims[1][1],
                     trims[2][0]:shp[2] - trims[2][1]
                     ]
            curr_y = curr_y[
                     trims[0][0]:shp[0] - trims[0][1],
                     trims[1][0]:shp[1] - trims[1][1],
                     trims[2][0]:shp[2] - trims[2][1]
                     ]
            pass
        # Down-sample the image now
        if ds_factor != 1:
            curr_x = curr_x[0::ds_factor, 0::ds_factor, 0::ds_factor]
            curr_y = curr_y[0::ds_factor, 0::ds_factor, 0::ds_factor]
        print('New shape: {}'.format(curr_x.shape)) if verbose else None
        ################################################################################################################
        ################################################################################################################
        ################################################################################################################
        # Add a dimension
        curr_x = curr_x[..., None]
        curr_y = curr_y[..., None]
        # Print Data Type and Shape Info
        # print("X/Y dtype {} / {}".format(curr_x.dtype, curr_y.dtype))
        # print("X/Y shape {} / {}".format(curr_x.shape, curr_y.shape))
        # Save up-dimensioned X (image file) to processed folder
        save_as_nifti(curr_x, output_data_dirpath, datum[0].replace('_LFOV.nii.gz', '') \
                      + '_AUG_0_ORIG' + '_LFOV.nii.gz')
        # Convert Y (segmented file) to Binary Class (0/1 Prostate) and save to processed folder
        # curr_y = simplify_labels(curr_y)
        save_as_nifti(curr_y, output_data_dirpath, datum[1].replace('_SEMANTIC_LFOV.nii.gz', '') \
                      + '_AUG_0_ORIG' + '_SEMANTIC_LFOV.nii.gz')
        # ####################################
        # Begin Augmentation
        # ####################################
        for aug_num in range(aug_count):
            # Create augmenter
            aug = ImageSegmentationAugmenter()
            aug_type_str = ''
            aug_type_int = np.random.randint(0,3)  # 0 = Flip, 1 = Warp, 2 = Both
            # Pick the augmentation type(s) randomly via a coin toss
            if aug_type_int in (0, 2):
                # Apply a flip
                aug.add_augmentation(Flip(1))
                aug_type_str = aug_type_str + 'FLIP'
            if aug_type_int in (1, 2):
                # Apply a warp
                aug.add_augmentation(GridWarp())
                aug_type_str = aug_type_str + 'WARP'
        # Filename manipulation
            save_name_x = datum[0].replace('_LFOV.nii.gz', '') \
                          + '_AUG_{}_{}'.format(aug_num + 1, aug_type_str) + '_LFOV.nii.gz'
            save_name_y = datum[1].replace('_SEMANTIC_LFOV.nii.gz', '') \
                          + '_AUG_{}_{}'.format(aug_num + 1, aug_type_str) + '_SEMANTIC_LFOV.nii.gz'
            # Augmentation is applied to X and Y simultaneously to ensure sync
            curr_x_aug, curr_y_aug = aug([curr_x, curr_y])
            # Convert to Numpy Float32 Arrays
            curr_x_aug = np.array(curr_x_aug, dtype=np.float32)
            curr_y_aug = np.array(curr_y_aug, dtype=np.float32)
            # Round to 0 decimals to match original data
            curr_x_aug = np.round(curr_x_aug, decimals=0)
            curr_y_aug = np.round(curr_y_aug, decimals=0)
            # Save augmented X and Y
            save_as_nifti(curr_x_aug, output_data_dirpath, save_name_x)
            save_as_nifti(curr_y_aug, output_data_dirpath, save_name_y)
            pass
    pass


def main():
    print('Augmenter Environment: {}={}, {}={}'.format('np', np.__version__, 'tf', tf.__version__))
    # Define the directories
    orig_data_dir = ''
    x_subdir = '/semantic_MRs_anon'
    y_subdir = '/semantic_labels_anon'
    output_dir = orig_data_dir + '/processed_downsampled'
    # Augment!
    data_preprocess_augment(orig_data_dir, x_subdir, y_subdir, output_dir, ds_factor=2, aug_count=0, verbose=True)
    pass


# Run
main()