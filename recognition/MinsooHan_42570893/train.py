import dataset
import modules
import utils
import tensorflow as tf
import SimpleITK as itk
import numpy as np
import pandas as pd
from skimage import io
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam

#data_processing = dataset.data_preprocessing()

save_train_data = 'E:/Uni/COMP3710/Assignment\PatternFlow/recognition/MinsooHan_42570893/train_data/'
save_training_ground_truth_data = 'E:/Uni/COMP3710/Assignment/PatternFlow/recognition/MinsooHan_42570893/train_ground_truth_data/'
save_validation_data = 'E:/Uni/COMP3710/Assignment/PatternFlow/recognition/MinsooHan_42570893/validation_data/'
save_validation_ground_truth_data = 'E:/Uni/COMP3710/Assignment/PatternFlow/recognition/MinsooHan_42570893/validation_ground_truth_data/'
save_test_data = 'E:/Uni/COMP3710/Assignment/PatternFlow/recognition/MinsooHan_42570893/test_data/'
save_test_ground_truth_data = 'E:/Uni/COMP3710/Assignment/PatternFlow/recognition/MinsooHan_42570893/test_ground_truth_data/'


# Load metadata.csv files of training and test data. And also load and store the images and masks in the training and test directories.
metadata_training = pd.read_csv('E:/Uni/COMP3710/ISIC-2017_Training_Data_metadata.csv')
metadata_validation = pd.read_csv('E:/Uni/COMP3710/ISIC-2017_Validation_Data_metadata.csv')
metadata_test = pd.read_csv('E:/Uni/COMP3710/ISIC-2017_Test_v2_Data_metadata.csv')



def train_data():
    x , y = [], []
    for index, cell in metadata_training.iterrows():
        read_image = itk.ReadImage(save_train_data + cell[0] + '.png')
        image_array = itk.GetArrayFromImage(read_image) / 255.0
        mask = io.imread(save_training_ground_truth_data + cell[0] + '_segmentation.png') / 255.0
        x.append(image_array)
        y.append(mask)
    return x, y

def validation_data():
    x , y = [], []
    for index, cell in metadata_validation.iterrows():
        read_image = itk.ReadImage(save_validation_data + cell[0] + '.png')
        image_array = itk.GetArrayFromImage(read_image) / 255.0
        mask = io.imread(save_validation_ground_truth_data + cell[0] + '_segmentation.png') / 255.0
        x.append(image_array)
        y.append(mask)
    return x, y

def test_data():
    x, y = [], []
    for index, cell in metadata_test.iterrows():
        read_image = itk.ReadImage(save_test_data + cell[0] + '.png')
        image_array = itk.GetArrayFromImage(read_image) / 255.0
        mask = io.imread(save_test_ground_truth_data + cell[0] + '_segmentation.png') / 255.0
        x.append(image_array)
        y.append(mask)
    return x, y


train_x, train_y = train_data()
validation_x, validation_y = validation_data()
test_x, test_y = test_data()

train_x = np.asarray(train_x)
train_y = np.asarray(train_y)
train_y1 = train_y
train_y = np.expand_dims(train_y, axis=-1)
z = np.zeros(train_y.shape[:-1] + (2,), dtype=train_y.dtype)
train_y = np.concatenate((train_y, z), axis=-1)

validation_x = np.asarray(validation_x)
validation_y = np.asarray(validation_y)
validation_y = np.expand_dims(validation_y, axis=-1)
z2 = np.zeros(validation_y.shape[:-1] + (2,), dtype=validation_y.dtype)
validation_y = np.concatenate((validation_y, z2), axis=-1)

test_x = np.array(test_x)
test_y= np.array(test_y)
test_y = np.expand_dims(test_y, axis=-1)
z3 = np.zeros(test_y.shape[:-1] + (2,), dtype=test_y.dtype)
test_y = np.concatenate((test_y, z3), axis=-1)

input_image = Input((256, 256, 3))
model = modules.improved_Unet(input_image)
learning_rate = 0.0005
decay_rate = learning_rate * 0.985
model.compile(optimizer=Adam(learning_rate=learning_rate, decay=decay_rate), loss=utils.dice_coef_loss, metrics=[utils.dice_coef])
model.summary()
model.fit(x=train_x, y=train_y, validation_data=(validation_x, validation_y), batch_size=8, epochs=300)
model.save('improvedUnet.h5')

