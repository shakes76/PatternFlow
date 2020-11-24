"""
Classify laterality of  the OAI AKOA with minimum accuracy of 0.9
@author : Wangwenhao(46039941)
"""
import os
import shutil
import numpy as np
from model import *
import matplotlib.pyplot as plt
from tensorflow.keras import models
from sklearn.metrics import classification_report

# parameters

image_height = 224
image_width = 224
batch_size = 32

# split data, preprocess the data
def data_split():
    # split the original image into train,test,val

    files = np.array(os.listdir("AKOA_Analysis"))
    index = [i for i in range(len(files))]
    np.random.shuffle(index)
    train_end = int(len(files) * 0.7)
    val_end = int(len(files) * 0.9)
    train_files = files[:train_end].tolist()
    val_files = files[train_end:val_end].tolist()
    test_files = files[val_end:].tolist()

    # copy and split the data into left and right part according to their name

    old_dir = "AKOA_Analysis"
    train_right_dir = "train/right"
    train_left_dir = "train/left"
    val_right_dir = "val/right"
    val_left_dir = "val/left"
    test_right_dir = "test/right"
    test_left_dir = "test/left"
    for file in train_files:
        old_file_path = os.path.join(old_dir, file)
        if file.split("_WE_")[1][0] == 'R':
            new_file_path = os.path.join(train_right_dir, file)
        if file.split("_WE_")[1][0] == 'L':
            new_file_path = os.path.join(train_left_dir, file)
        shutil.copy(old_file_path, new_file_path)

    for file in val_files:
        old_file_path = os.path.join(old_dir, file)
        if file.split("_WE_")[1][0] == 'R':
            new_file_path = os.path.join(val_right_dir, file)
        if file.split("_WE_")[1][0] == 'L':
            new_file_path = os.path.join(val_left_dir, file)
        shutil.copy(old_file_path, new_file_path)

    for file in test_files:
        old_file_path = os.path.join(old_dir, file)
        if file.split("_WE_")[1][0] == 'R':
            new_file_path = os.path.join(test_right_dir, file)
        if file.split("_WE_")[1][0] == 'L':
            new_file_path = os.path.join(test_left_dir, file)
        shutil.copy(old_file_path, new_file_path)


# plot and visualize the detail

def visualise(model):
    # visualise accuracy of training
    plt.plot(model.history.history['accuracy'])
    plt.plot(model.history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    # visualise loss of training
    plt.plot(model.history.history['loss'])
    plt.plot(model.history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


# predict and show the precision and recall
def predict(model):
    data_generator = preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    test_generator = data_generator.flow_from_directory(
        model.test_path,
        target_size=(model.image_height, model.image_width),
        batch_size=1,
        seed=0)

    pred = model.model.predict_generator(test_generator)
    predicted_class_indices = np.argmax(pred, axis=1)
    true_label = test_generator.classes
    print(classification_report(true_label, predicted_class_indices))


if __name__ == '__main__':
    data_split()
    kneeModel = KneeModel("train", "val", "test")
    kneeModel.fit()
    # kneeModel.load_model('model.h5')
    kneeModel.evaluate()
    visualise(kneeModel)
    predict(kneeModel)
