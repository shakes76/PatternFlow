
import numpy as np
import os
import cv2
import random


class LoadData:

    def __init__(self):
        self.load_training_data()
        self.load_validation_data()
        self.load_testing_data()

    def load_training_data(self):
        DATADIR = "ADNI_AD_NC_2D/AD_NC/train"
        CATEGORIES = ["AD", "NC"]
        count = 0
        training_data = []
        for category in CATEGORIES:
            path = os.path.join(DATADIR, category)
            class_num = CATEGORIES.index(category)
            for img in os.listdir(path):
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                training_data.append([img_arr, class_num])
                count += 1
                if (count == 19368):
                    break                    
            if (count == 19368):
                    break
        print("Loaded " + str(count) + " train images", flush=True)
        random.shuffle(training_data)
        
        X_train = []
        y_train = []
        
        for features, label in training_data:
            X_train.append(features)
            y_train.append(label)
        
        X_train = np.array(X_train).reshape(-1, 240, 256, 1)
        print(X_train.shape)

        np.save("X_train.npy", X_train)
        np.save("y_train.npy", y_train)


    def load_validation_data(self):
        DATADIR = "ADNI_AD_NC_2D/AD_NC/train"
        CATEGORIES = ["AD", "NC"]
        count = 0
        training_data = []
        for category in CATEGORIES:
            path = os.path.join(DATADIR, category)
            class_num = CATEGORIES.index(category)
            for img in os.listdir(path):
                count += 1
                if (count <= 19368):
                    continue
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                training_data.append([img_arr, class_num])

        print("Loaded " + str(count - 19368) + " validation images", flush=True)
        random.shuffle(training_data)
        
        X = []
        y = []
        
        for features, label in training_data:
            X.append(features)
            y.append(label)
        
        X = np.array(X).reshape(-1, 240, 256, 1)
        print(X.shape)

        np.save("X_validate.npy", X)
        np.save("y_validate.npy", y)

    def load_testing_data(self):
        DATADIR = "ADNI_AD_NC_2D/AD_NC/test"
        CATEGORIES = ["AD", "NC"]
        count = 0
        training_data = []
        for category in CATEGORIES:
            path = os.path.join(DATADIR, category)
            class_num = CATEGORIES.index(category)
            for img in os.listdir(path):
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                training_data.append([img_arr, class_num])
                count += 1
                
        print("Loaded " + str(count) + " test images", flush=True)
        random.shuffle(training_data)
        
        X_train = []
        y_train = []
        
        for features, label in training_data:
            X_train.append(features)
            y_train.append(label)
        
        X_train = np.array(X_train).reshape(-1, 240, 256, 1)
        print(X_train.shape)

        np.save("X_test.npy", X_train)
        np.save("y_test.npy", y_train)