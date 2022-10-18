import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array

#input_dir = 'ISIC-2017_Training_Data/ISIC-2017_Training_Data'
#target_dir = 'ISIC-2017_Training_Part1_GroundTruth/ISIC-2017_Training_Part1_GroundTruth'

class Dataloader:
    def __init__(self, img_path, mask_path, IMG_WIDTH=1022, IMG_HEIGHT=767, split=0.8):
            self.IMG_WIDTH = IMG_WIDTH
            self.IMG_HEIGHT = IMG_HEIGHT
            self.img_path = img_path
            self.mask_path = mask_path
            self.split = split

    def load_image_path(self, img, ground):
        input_img_paths = sorted(
            [
                os.path.join(img, fname)
                for fname in os.listdir(input_dir)
                if fname.endswith(".jpg")
            ]
        )
        target_img_paths = sorted(
            [
                os.path.join(ground, fname)
                for fname in os.listdir(target_dir)
                if fname.endswith(".png")
            ]
        )
        return input_img_paths, target_img_paths

    def process_image(self, IMG_WIDTH, IMG_HEIGHT, img_path, mask_path):
        X = np.zeros((len(img_path), IMG_WIDTH, IMG_HEIGHT, 3), dtype=np.float32)
        Y = np.zeros((len(img_path), IMG_WIDTH, IMG_HEIGHT, 1), dtype=np.uint8)

        for i in range(len(img_path)):
            img = img_path[i]
            img = load_img(img, color_mode="rgb", target_size=(IMG_WIDTH, IMG_HEIGHT)) 
            img = img_to_array(img)
            X[i] = img.astype('float32') / 255.0
            
            mask = mask_path[i]
            mask = load_img(mask, color_mode="grayscale", target_size=(IMG_WIDTH, IMG_HEIGHT))
            mask = img_to_array(mask)
            Y[i] = mask
        return X,Y

    def split_data(self, img_path, X, Y, split):
        train_test_split = int(len(img_path)*split)
        X_train = X[:train_test_split]
        Y_train = Y[:train_test_split]
        X_test = X[train_test_split:]
        Y_test = Y[train_test_split:]
        return X_train, Y_train, X_test, Y_test

    def get_XY_split(self):
        input, target = self.load_image_path(self, self.img_path, self.mask_path)
        X,Y = self.process_image(self.IMG_WIDTH, self.IMG_HEIGHT, input, target)
        X_train, Y_train, X_test, Y_test = self.split_data(input, X,Y, self.split)
        return X_train, Y_train, X_test, Y_test



