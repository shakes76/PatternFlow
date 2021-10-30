import fnmatch
import os
import nibabel as nib
import numpy as np
import pandas as pd
import datetime
import shutil

import tensorflow.keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

import model as m
from model import UNetCSIROMalePelvic

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


def read_and_split_data(processed_data_dir, split_perc_arr, split_names_arr, verbose=False, rng_seed=None):
    """Given a processed* data directory and a split regime (percs and names), returns a Pandas DataFrame
    containing all details about the dataset (filepaths, split bucket etc.). Each row is for an X/Y file pair.
    Splits can be deterministic if an RNG Seed is provided (optional).
     \n*Expects the folder to contain the output of the data_preprocess_augment() function."""

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
        actual_splits[i: i + curr_split_count] = str(split_names_arr[idx])
        i += curr_split_count
        pass
    # Randomize (with seed, if provided)
    np.random.seed(seed=rng_seed) if rng_seed is not None else None
    np.random.shuffle(actual_splits)
    # Enforce correct length in case of round-up errors
    actual_splits = actual_splits[0:num_unique_cases]
    actual_splits = pd.DataFrame(actual_splits, columns=['split_type'])
    # Join with unique cases
    unique_cases = pd.merge(unique_cases, actual_splits, how='inner', left_index=True, right_index=True)
    # Join back with main dataframe
    data_files = pd.merge(data_files, unique_cases, how='inner', left_on='case_num', right_on='case_num')
    return data_files


def prep_sample_for_training(x, y, num_classes):
    """Mandatory pre-processing tasks before feeding a datum to the UNetCSIROMalePelvic model.
    Takes a SINGLE x and y sample and processes it."""
    # Standardize the X data (Mean 0, Unit Variance)
    x = standardize(x)
    # One-hot Encode Y label
    y = to_categorical(y, num_classes=num_classes, dtype=np.int32)
    # Add a dimension at the start for Batch Size of 1
    x = x[None, ...]
    y = y[None, ...]
    return x, y

# Evaluate the model's DSC score on the entire dataset of the given split type
def evaluate_dsc_model(model_class, master_df, split_type, num_classes, return_summary=True, save_preds=False):
    """Evaluate the DSC on the UNetCSIROMalePelvic for the given data frame / data type (val, test etc.)."""
    scores = []
    for idx, row in master_df[master_df['split_type'] == split_type].iterrows():
        # print(row.case_num, row.observation_num, row.x_filename)

        # Skip augmented data for validation / test
        # Augmented data is only for training
        if split_type in ('val', 'test'):
            if str(row.aug_type) != 'ORIG':
                continue

        # print(row.case_num, row.observation_num, row.x_filename)

        # Load the X Y files
        curr_x = nib.load(row.x_filepath).get_fdata()
        curr_y = nib.load(row.y_filepath).get_fdata()

        # Pre-process for training
        curr_x, curr_y = prep_sample_for_training(x=curr_x, y=curr_y, num_classes=num_classes)

        # Predict and score (quietly)
        prediction = model_class.mdl.predict(curr_x)
        score = m.f1_score(y_true=curr_y, y_pred=prediction, return_list=True, verbose=False)
        scores.append(score)

        if save_preds:
            # Argmax from 6x class probability matrices --> Single final class matrix per voxel
            prediction_max = np.argmax(prediction, axis=-1)
            #print('prediction_max shape:', prediction_max.shape)
            #np.unique(prediction_max, return_counts=True)
            prediction_max = np.array(prediction_max[0][..., None], dtype=np.float64)
            # Save it
            output_data_dirpath = os.path.dirname(row.x_filepath) + '/' + model_class.mdl.name + '_predictions/'

            # Make the output directory if needed
            if not os.path.exists(output_data_dirpath):
                os.mkdir(output_data_dirpath)

            save_as_nifti(prediction_max, output_data_dirpath, row.x_filename + '_pred.nii.gz')
            # Save a copy of the orig label file alongside
            shutil.copy(row.y_filepath, output_data_dirpath)

        pass
    # Return Mean DSC for each class across all the samples tested unless summary is False
    return np.mean(scores, axis=0) if return_summary else scores
    pass


def train():
    val_required = True
    split_rng_seed = 12345
    num_classes = 6
    train_loops = 30000
    val_frequency = 1000
    print(np.__version__)
    cleaned_data_dir = ''

    # Split the data
    if val_required:
        # Create Train / Val / Test splits
        split_perc_arr = np.array([0.6, 0.2, 0.2], dtype=np.float32)
        split_names_arr = np.array(['train', 'val', 'test'])
    else:
        # Create Train / Test splits
        split_perc_arr = np.array([0.8, 0.2], dtype=np.float32)
        split_names_arr = np.array(['train', 'test'])

    master_df = read_and_split_data(cleaned_data_dir, split_perc_arr, split_names_arr, rng_seed=split_rng_seed)
    print(master_df.info(verbose=True))

    # Create the model
    the_model = UNetCSIROMalePelvic(given_name="LabPCAugScale1", num_classes=num_classes, feature_map_scale=1)
    print(the_model.mdl.summary(line_length=200))
    for layer in the_model.mdl.layers:
        print(layer.input_shape, '-->', layer.name, '-->', layer.output_shape)
        pass

    # Main Training Loop
    for i in range(train_loops):
        print("{} | Start Train Loop {} of {} | Total Exp: {}".format(datetime.datetime.now(), i, train_loops,
                                                                      the_model.train_batch_count))
        # Collect a training sample
        sample = master_df[master_df['split_type'] == 'train'].sample(n=1)
        # Load the X Y files
        curr_x = nib.load(sample.x_filepath.item()).get_fdata()
        curr_y = nib.load(sample.y_filepath.item()).get_fdata()

        # Pre-process for training
        curr_x, curr_y = prep_sample_for_training(x=curr_x, y=curr_y, num_classes=num_classes)

        # Do the training
        result = the_model.mdl.train_on_batch(x=curr_x, y=curr_y, reset_metrics=False, return_dict=True)
        #result = the_model.mdl.fit(x=curr_x, y=curr_y, batch_size=1, epochs=1, verbose='auto',
        #                           callbacks=[the_model.CustomCallBack()], validation_data=None, shuffle=False)
        # Increment the trained batch count for the model class (self managed)
        the_model.train_batch_count += 1
        print('{} | Result: {}'.format(datetime.datetime.now(), result))

        # Evaluate performance and save the model every now and then
        if i > 0 and i % val_frequency == 0:
            print('Starting Model Evaluation...')
            evaluation = evaluate_dsc_model(model_class=the_model, master_df=master_df, split_type='val',
                                            num_classes=num_classes, return_summary=True)
            print('Model Evaluation:', evaluation)
            print('Saving Model...')
            the_model.save_model(suffix_note='Seed12345', loc=r"")
        pass
    pass


# Test a saved model. Ensure the seed for data splitting is the same as the one used to train the model!
def test_saved_model():
    """Loads a saved model which has been trained previously, and evaluates it against a test set.
    User to ensure the test set is not in conflict with the train / val set the model was trained on originally,
    otherwise the test score is not accurate and the model will be cheating."""
    val_required = True
    split_rng_seed = 12345
    num_classes = 6
    print(np.__version__)
    cleaned_data_dir = ''

    # Split the data
    if val_required:
        # Create Train / Val / Test splits
        split_perc_arr = np.array([0.6, 0.2, 0.2], dtype=np.float32)
        split_names_arr = np.array(['train', 'val', 'test'])
    else:
        # Create Train / Test splits
        split_perc_arr = np.array([0.8, 0.2], dtype=np.float32)
        split_names_arr = np.array(['train', 'test'])

    master_df = read_and_split_data(cleaned_data_dir, split_perc_arr, split_names_arr, rng_seed=split_rng_seed)
    print(master_df.info(verbose=True))


    tf_model = load_model(r'',
                          custom_objects={'f1_score': m.f1_score})

    # Re-create the outer class holding the TF model and assign the loaded TF model to it
    the_model = UNetCSIROMalePelvic(given_name=tf_model.name, num_classes=num_classes)
    the_model.mdl = tf_model

    scores = evaluate_dsc_model(model_class=the_model, master_df=master_df, split_type='test',
                       num_classes=num_classes, return_summary=True, save_preds=True)
    print('evaluation:', scores)

# Run the program
train()
#test_saved_model()


