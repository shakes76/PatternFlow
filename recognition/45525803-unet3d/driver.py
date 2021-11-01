from process_data import get_case_weeks, write_original_and_augmented
from model import unet3d_model, MRISequence

import tensorflow as tf
import math
import os


MRI_PATH = '/home/Student/s4552580/mri_data/semantic_MRs_anon'
LABEL_PATH = '/home/Student/s4552580/mri_data/semantic_labels_anon'

CHECKPOINT_PATH = '/home/Student/s4552580/unet3d2.ckpt'
HISTORY_PATH = '/home/Student/s4552580/history2.csv'

TRAIN_CASE_NUMBERS = range(4,35)

TRAIN_VAL_RATIO = 0.8

def main():

    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

    physical_devices = tf.config.list_physical_devices('GPU') 
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    
    train_case_weeks = get_case_weeks(TRAIN_CASE_NUMBERS)

    mri_paths = [os.path.join(MRI_PATH, f'{x}_LFOV.nii.gz') for x in train_case_weeks]
    label_paths = [os.path.join(LABEL_PATH, f'{x}_SEMANTIC_LFOV.nii.gz') for x in train_case_weeks]

    train_val_split_index = math.ceil(len(mri_paths) * TRAIN_VAL_RATIO)

    train_mri_paths = mri_paths[:train_val_split_index]
    train_label_paths = label_paths[:train_val_split_index]
    train_seq = MRISequence(train_mri_paths, train_label_paths)

    val_mri_paths = mri_paths[train_val_split_index:]
    val_label_paths = label_paths[train_val_split_index:]
    val_seq = MRISequence(val_mri_paths, val_label_paths)
    
    model = unet3d_model()

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH,
                                                    save_weights_only=True,
                                                    verbose=1)

    history_callback = tf.keras.callbacks.CSVLogger(HISTORY_PATH, separator=",", append=True)

    history = model.fit(
        x=train_seq, 
        validation_data=val_seq, 
        callbacks=[cp_callback, history_callback],
        epochs=13)

if __name__ == '__main__':
    main()