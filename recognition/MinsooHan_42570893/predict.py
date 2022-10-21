import os
import SimpleITK as itk
import numpy as np
import pandas as pd
from skimage import io
from tensorflow.keras.models import load_model

save_test_data = 'E:/Uni/COMP3710/Assignment/PatternFlow/recognition/MinsooHan_42570893/test_data/'
save_test_ground_truth_data = 'E:/Uni/COMP3710/Assignment/PatternFlow/recognition/MinsooHan_42570893/test_ground_truth_data/'
metadata_test = pd.read_csv('E:/Uni/COMP3710/ISIC-2017_Test_v2_Data_metadata.csv')

def test_data():
    x, y = [], []
    for index, cell in metadata_test.iterrows():
        read_image = itk.ReadImage(save_test_data + cell[0] + '.png')
        image_array = itk.GetArrayFromImage(read_image) / 255.0
        mask = io.imread(save_test_ground_truth_data + cell[0] + '_segmentation.png') / 255.0
        x.append(image_array)
        y.append(mask)
    return x, y

test_x, test_y = test_data()
test_x = np.array(test_x)
test_y= np.array(test_y)
test_y = np.expand_dims(test_y, axis=-1)
z3 = np.zeros(test_y.shape[:-1] + (2,), dtype=test_y.dtype)
test_y = np.concatenate((test_y, z3), axis=-1)
