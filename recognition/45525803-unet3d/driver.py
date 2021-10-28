from process_data import get_case_weeks, write_original_and_augmented
from model import unet3d_model, MRISequence

import tensorflow as tf
import math
import os

PROCESSED_MRI_PATH = '/home/Student/s4552580/mri_data/processed_mri'
PROCESSED_LABEL_PATH = '/home/Student/s4552580/mri_data/processed_label'

TRAIN_CASE_NUMBERS = range(4,25)
TEST_CASE_NUMBERS = range(30,43)

TRAIN_VAL_RATIO = 0.85

def main():

    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

    physical_devices = tf.config.list_physical_devices('GPU') 
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    
    train_case_weeks = get_case_weeks(TRAIN_CASE_NUMBERS)
    mri_filenames, label_filenames = write_original_and_augmented(train_case_weeks, PROCESSED_MRI_PATH, PROCESSED_LABEL_PATH, 0)

    mri_paths = [os.path.join(PROCESSED_MRI_PATH, x) for x in mri_filenames]
    label_paths = [os.path.join(PROCESSED_LABEL_PATH, x) for x in label_filenames]

    train_val_split_index = math.ceil(len(mri_paths) * TRAIN_VAL_RATIO)

    train_mri_paths = mri_paths[:train_val_split_index]
    train_label_paths = label_paths[:train_val_split_index]
    train_seq = MRISequence(train_mri_paths, train_label_paths)

    val_mri_paths = mri_paths[train_val_split_index:]
    val_label_paths = label_paths[train_val_split_index:]
    val_seq = MRISequence(val_mri_paths, val_label_paths)
    
    model = unet3d_model()
    print(model.summary(line_length=150))

    history = model.fit(x=train_seq, validation_data=val_seq, epochs=5)

if __name__ == '__main__':
    main()