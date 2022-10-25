import numpy as np
import cv2
import os 
import sklearn
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')


DATADIR_train = './ADNI_AD_NC_2D/AD_NC/train'
DATADIR_test = './ADNI_AD_NC_2D/AD_NC/test'
classes = ['AD','NC']
training_data = []
testing_data = []
image_size = 512







def createTrainData(img_size,batch_size):
    train_ds = keras.utils.image_dataset_from_directory(
    directory=DATADIR_train,
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=(img_size, img_size)
    )
    return train_ds


    # for cur_class in classes:
    #     path = os.path.join(DATADIR_train,cur_class) # path to AD or NC dir
    #     class_num = classes.index(cur_class)
    #     for img in os.listdir(path):
    #         img_array = cv2.imread(os.path.join(path,img))
    #         training_data.append([img_array,class_num])

def createTestData(img_size,batch_size):
    test_ds = keras.utils.image_dataset_from_directory(
    directory=DATADIR_test,
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=(img_size, img_size)
    )
    return test_ds
    # for cur_class in classes:
    #     path = os.path.join(DATADIR_test,cur_class) # path to AD or NC dir
    #     class_num = classes.index(cur_class)
    #     for img in os.listdir(path):
    #         img_array = cv2.imread(os.path.join(path,img))
    #         #resize_array = cv2.resize(img_array,(img_size,img_size))
    #         testing_data.append([img_array,class_num])

# def prepareData():
#     createTrainData(img_size)
#     createTestData(img_size)
#     training = sklearn.utils.shuffle(training_data,random_state=77)
#     testing = sklearn.utils.shuffle(testing_data,random_state=77)
#     x_train = np.array([training[i][0] for i in range(len(training_data))])
#     y_train = np.array([training[i][1] for i in range(len(training_data))])
#     x_test = np.array([testing[i][0] for i in range(len(testing_data))])
#     y_test = np.array([testing[i][1] for i in range(len(testing_data))])
#     print(len(y_test))
#     return x_train, y_train, x_test, y_test
