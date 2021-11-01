import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import img_to_array
from perceiver import Perceiver, fitModel
import os
from tensorflow.keras.preprocessing.image import load_img
import numpy as np

def generate_samples(instanceIds, dataset_proportion):
    """Split data into training and testing set.
    Note: Imgs of the same person are not in both sets
    """
    samples = dict()
    trainIndex = []
    testIndex = []
    for fileId in instanceIds:
        fileName = fileId.split('_')[0]
        if fileName in samples:
            samples[fileName].append(fileId)
        else:
            samples[fileName] = [fileId]
    completedTrainSet = False #flag for processingtrain set
    for imgIndex in samples.values():
        for fileId in imgIndex:
            if not completedTrainSet:
                if dataset_proportion * len(instanceIds) < len(trainIndex):
                    completedTrainSet = True
                    break
                else:
                    trainIndex.append(fileId)
            else:
                if (1 - dataset_proportion) * len(instanceIds) >= len(testIndex):
                    testIndex.append(fileId)
                else:
                    break
    return trainIndex, testIndex

def loadInput(dir_data, dataset_proportion):
    """Initialize tensors for xtrain, ytrain, xtest, ytest"""
    #18680 instances
    all_instanceId = os.listdir(dir_data)
    trainIndex, testIndex = generate_samples(all_instanceId, dataset_proportion)
    # random.shuffle(trainIndex)
    # random.shuffle(testIndex)

    def load_samples(instanceIds):
        #load tensor X, y from images
        input = []
        list_of_labels = []
        for i, fileId in enumerate(instanceIds): #label
            image = load_img(dir_data + "/" + fileId, target_size=((64, 64)), color_mode="grayscale")
            image = img_to_array(image)
            input.append(image)
            if "Left" in fileId or "LEFT" in fileId or "left" in fileId or "L_E_F_T" in fileId:
                label = 0
            else:
                label = 1
            list_of_labels.append(label)
        input = np.array(input)
        input /= 255.0
        return input, np.array(list_of_labels, dtype=np.uint8).flatten()

    trainX, trainY = load_samples(trainIndex)
    valX, valY = load_samples(testIndex)
    testX, testY = valX[len(valX) // 5 * 4:], valY[len(valY) // 5 * 4:]
    valX, valY = valX[0:len(valX) // 5 * 4], valY[0:len(valY) // 5 * 4]
    
    return trainX, trainY, valX, valY, testX, testY

def create_dataset(image_dir, img_size):
    """Create train, val, test dataset"""
    split_train = image_dataset_from_directory(image_dir, validation_split=0.2, color_mode='grayscale',
                                                    subset="training", seed=46, label_mode='int',
                                                    batch_size=1, image_size=img_size)
    split_val = image_dataset_from_directory(image_dir, validation_split=0.2, color_mode='grayscale',
                                                      subset="validation", seed=46, label_mode='int',
                                                      batch_size=1, image_size=img_size)
    # Split val set to into test set
    foldSize = tf.data.experimental.cardinality(split_val)
    split_test = split_val.take(foldSize // 5)
    split_val = split_val.skip(foldSize // 5)
    #shuffle Imgs
    split_train = split_train.map(process).prefetch(AUTO_TUNE)
    split_val = split_val.map(process).prefetch(AUTO_TUNE)
    split_test = split_test.map(process).prefetch(AUTO_TUNE)
    return split_train, split_val, split_test

def process(inputImg, groundTruth):
    """Normalize Image"""
    inputImg = tf.cast(inputImg / 255. ,tf.float32)
    return inputImg, groundTruth

if __name__ == "__main__":

    dataset_proportion = 0.8
    trainX, trainY, valX, valY, testX, testY = loadInput('./AKOA_Analysis', dataset_proportion)
    print(len(trainX), len(trainY), len(valX), len(valY), len(testX), len(testY))
    print(trainX.shape)

    perceiverTransformer = Perceiver(inDim=64*64, latentDim=256, freq_ban=4, proj_size=19, max_freq=10)

    history = fitModel(perceiverTransformer, train_set=(trainX, trainY),
            val_set=(valX, valY), test_set=(testX, testY), batch_size=32)
