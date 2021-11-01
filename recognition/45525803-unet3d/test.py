"""
Author: Joshua Knowles
Student ID: 45525803
Date: 30/10/2021

Tests the 3D U-Net model on the test data given the latest checkpoint.
"""

from process_data import get_case_weeks
from model import unet3d_model, dice_coefficient, MRISequence
from pyimgaug3d.utils import to_channels

import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
import nibabel as nib

MRI_PATH = '/home/Student/s4552580/mri_data/semantic_MRs_anon'
LABEL_PATH = '/home/Student/s4552580/mri_data/semantic_labels_anon'

CHECKPOINT_PATH = '/home/Student/s4552580/unet3d.ckpt'

IMAGES_PATH = '/home/Student/s4552580/images'

TEST_CASE_NUMBERS = range(35,43)

if __name__ == '__main__':

    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

    physical_devices = tf.config.list_physical_devices('GPU') 
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

    test_case_weeks = get_case_weeks(TEST_CASE_NUMBERS)

    test_mri_paths = [os.path.join(MRI_PATH, f'{x}_LFOV.nii.gz') for x in test_case_weeks]
    test_label_paths = [os.path.join(LABEL_PATH, f'{x}_SEMANTIC_LFOV.nii.gz') for x in test_case_weeks]

    test_seq = MRISequence(test_mri_paths, test_label_paths)

    checkpoint_dir = os.path.dirname(CHECKPOINT_PATH)
    latest_cp = tf.train.latest_checkpoint(checkpoint_dir)

    print(f'Latest checkpoint: {latest_cp}')

    model = unet3d_model()
    model.load_weights(latest_cp)

    print('Predicting..')

    out = model.predict(test_seq)
    seg_preds = np.round(out)
    seg_preds = tf.convert_to_tensor(seg_preds, dtype=tf.float32)
    label_preds = np.argmax(seg_preds, axis=-1).astype('int16')

    print('Evaluating..')

    test_dice_coeffs = np.zeros((len(test_case_weeks), 6))
    for i, label_path in enumerate(test_label_paths):

        seg_true = to_channels(nib.load(label_path).get_fdata())
        seg_true = tf.convert_to_tensor(seg_true, dtype=tf.float32)
        label_true = np.argmax(seg_true, axis=-1).astype('int16')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,10))
        ax1.imshow(label_true[128,:,:])
        ax1.set_title('True Labels')
        ax2.imshow(label_preds[i,128,:,:])
        ax2.set_title('Predicted Labels')
        plt.savefig(os.path.join(IMAGES_PATH, f'{i}.png'))

        for j in range(6):
            test_dice_coeffs[i,j] = dice_coefficient(seg_true[:,:,:,j], seg_preds[i,:,:,:,j])

    for j in range(6):
        print(f'\n=== Label {j+1} ===')
        print(f'Minimum DSC: {min(test_dice_coeffs[:,j])}')
        print(f'Maximum DSC: {max(test_dice_coeffs[:,j])}')
        print(f'Mean DSC: {test_dice_coeffs[:,j].mean()}')