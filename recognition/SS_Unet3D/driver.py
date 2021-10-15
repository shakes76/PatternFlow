
import fnmatch
import os
import nibabel as nib
import numpy as np
import pandas as pd

from model import UNetCSIROMalePelvic

from pyimgaug3d.augmentation import GridWarp, Flip
from pyimgaug3d.augmenters import ImageSegmentationAugmenter

# Credit: Taken from demo.py in pyimgaug3d
def save_as_nifti(data, folder, name, affine=np.eye(4)):
    img = nib.Nifti1Image(data, affine)
    if not os.path.exists(folder):
        os.mkdir(folder)
    nib.save(img, os.path.join(folder, name))
    pass

# Standardizes the NP array (0 Mean & Unit Variance)
def standardize(given_np_arr):
    return (given_np_arr - given_np_arr.mean()) / given_np_arr.std()

# Pre-process and Augment the CSIRO 3D Male Pelvic dataset
# Original data is expected in two sub-folders, but is outputted to a single directory
def data_preprocess_augment(orig_data_dirpath, orig_data_x_subdirname, orig_data_y_subdirname, output_data_dirpath,
                            verbose=False):
    aug_count = 3
    flip_augs = [0]

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

    # Make the output directory if needed
    if not os.path.exists(output_data_dirpath):
        os.mkdir(output_data_dirpath)

    # Lambda function to merge all classes except 5 (Prostate)
    simplify_labels = np.vectorize(lambda x: float(x == 5))

    # Process each image
    for idx, datum in enumerate(data_filenames):
        print('Augmenting observation #{} of {}: {}'.format(idx + 1, len(data_filenames), datum)) if verbose else None
        # Load files into memory
        curr_x = nib.load(orig_data_x_dirpath + '/' + datum[0]).get_fdata()
        curr_y = nib.load(orig_data_y_dirpath + '/' + datum[1]).get_fdata()
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
        curr_y = simplify_labels(curr_y)
        save_as_nifti(curr_y, output_data_dirpath, datum[1].replace('_SEMANTIC_LFOV.nii.gz', '') \
                      + '_AUG_0_ORIG' + '_SEMANTIC_LFOV.nii.gz')
        # ####################################
        # Begin Augmentation
        # ####################################
        for aug_num in range(aug_count):
            # Create augmenter
            aug = ImageSegmentationAugmenter()
            if aug_num in flip_augs:
                # Apply a flip on the specified Axis
                aug.add_augmentation(Flip(aug_num + 1))
                aug_type = 'FLIP'
            else:
                # Apply a warp
                aug.add_augmentation(GridWarp())
                aug_type = 'WARP'

            # Filename manipulation
            save_name_x = datum[0].replace('_LFOV.nii.gz', '') \
                          + '_AUG_{}_{}'.format(aug_num + 1, aug_type) + '_LFOV.nii.gz'
            save_name_y = datum[1].replace('_SEMANTIC_LFOV.nii.gz', '') \
                          + '_AUG_{}_{}'.format(aug_num + 1, aug_type) + '_SEMANTIC_LFOV.nii.gz'
            # Augmentation is applied to X and Y simultaneously to ensure sync
            curr_x_aug, curr_y_aug = aug([curr_x, curr_y])
            # Convert to Numpy Float32 Arrays
            curr_x_aug = np.array(curr_x_aug, dtype=np.float32)
            curr_y_aug = np.array(curr_y_aug, dtype=np.float32)
            # Standardize
            curr_x_aug = standardize(curr_x_aug)
            curr_y_aug = standardize(curr_y_aug)
            # Round to 0 decimals to match original data
            # curr_x_aug = np.round(curr_x_aug, decimals=0)
            # curr_y_aug = np.round(curr_y_aug, decimals=0)
            # Save augmented X and Y
            save_as_nifti(curr_x_aug, output_data_dirpath, save_name_x)
            save_as_nifti(curr_y_aug, output_data_dirpath, save_name_y)
            pass
    pass

def read_and_split_data(processed_data_dir, split_perc_arr, split_names_arr, verbose=False):
    x_files = np.array(fnmatch.filter(sorted(os.listdir(processed_data_dir)), '*.nii.gz'))
    x_files = x_files[np.array([not fnmatch.fnmatch(filename, '*SEMANTIC*') for filename in x_files])]
    y_files = np.array(fnmatch.filter(sorted(os.listdir(processed_data_dir)), '*SEMANTIC*.nii.gz'))
    print('#X:{}, #Y:{}'.format(len(x_files), len(y_files))) if verbose else None

    x_files = pd.DataFrame(x_files)
    x_files.columns = ['x_filename']
    x_files = pd.merge(x_files, x_files['x_filename'].str.split('_', 10, expand=True),
                       left_index=True, right_index=True)
    x_files.columns = ['x_filename', 'preamble', 'case_num', 'observation_num', 'aug', 'aug_num', 'aug_type', 'suffix']
    x_files = x_files[['preamble', 'case_num', 'observation_num', 'aug_num', 'aug_type', 'x_filename']]
    x_files['x_filepath'] = x_files['x_filename'].apply(lambda x: processed_data_dir + '/' + x)

    y_files = pd.DataFrame(y_files)
    y_files.columns = ['y_filename']
    y_files = pd.merge(y_files, y_files['y_filename'].str.split('_', 10, expand=True),
                       left_index=True, right_index=True)
    y_files.columns = ['y_filename', 'preamble', 'case_num', 'observation_num', 'aug', 'aug_num', 'aug_type',
                       'semantic', 'suffix']
    y_files = y_files[['preamble', 'case_num', 'observation_num', 'aug_num', 'aug_type', 'y_filename']]
    y_files['y_filepath'] = y_files['y_filename'].apply(lambda y: processed_data_dir + '/' + y)

    data_files = pd.merge(x_files, y_files, how='inner',
                          left_on=['preamble', 'case_num', 'observation_num', 'aug_num', 'aug_type'],
                          right_on=['preamble', 'case_num', 'observation_num', 'aug_num', 'aug_type'])

    # Distinct cases out of all successfully
    # merged X,Y pairs of images
    unique_cases = data_files['case_num'].unique()
    unique_cases = pd.DataFrame(unique_cases, columns=['case_num'])
    num_unique_cases = len(unique_cases)

    split_counts = np.round(split_perc_arr * num_unique_cases, decimals=0)
    split_counts = np.array(split_counts, dtype=np.int32)

    actual_splits = np.empty(num_unique_cases, dtype='<U11')
    i = 0
    for idx, curr_split_count in enumerate(split_counts):
        actual_splits[i: i + curr_split_count] = split_names_arr[idx]
        i += curr_split_count
        pass

    # Randomize
    np.random.shuffle(actual_splits)
    # Enforce correct length in case of round-up errors
    actual_splits = actual_splits[0:num_unique_cases]
    actual_splits = pd.DataFrame(actual_splits, columns=['split_type'])
    # Join with unique cases
    unique_cases = pd.merge(unique_cases, actual_splits, how='inner', left_index=True, right_index=True)

    # Join back with main dataframe
    data_files = pd.merge(data_files, unique_cases, how='inner', left_on='case_num', right_on='case_num')

    return data_files





def main():

    print(np.__version__)
    # need 1.18.5

    augment_required = False

    # Augment the dataset if required

    orig_data_dir = ''
    x_subdir = '/semantic_MRs_anon'
    y_subdir = '/semantic_labels_anon'
    output_dir = orig_data_dir + '/processed'

    if augment_required:
        data_preprocess_augment(orig_data_dir, x_subdir, y_subdir, output_dir, verbose=True)

    # Create the model
    the_model = UNetCSIROMalePelvic("My Model")

    print(the_model.mdl.summary())

    for layer in the_model.mdl.layers:
        print(layer.input_shape, '-->', layer.name, '-->', layer.output_shape)
        pass

    # Create Train / Val / Test splits
    split_perc_arr = np.array([0.4, 0.4, 0.2], dtype=np.float32)
    split_names_arr = np.array(['train', 'val', 'test'])
    master_df = read_and_split_data(output_dir, split_perc_arr, split_names_arr)

    print(master_df.info(verbose=True))

    # Collect a training sample
    sample = master_df[master_df['split_type'] == 'train'].sample(n=1)

    # Load it into memory
    curr_x = nib.load(sample.x_filepath.item()).get_fdata()
    curr_y = nib.load(sample.y_filepath.item()).get_fdata()

    print(curr_x.shape)
    print(curr_y.shape)

    the_model.mdl.train_on_batch(x=curr_x, y=curr_y)

    pass


# Run the program
main()