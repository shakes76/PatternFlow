import os
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array, get_file

#input_dir = 'ISIC-2017_Training_Data/ISIC-2017_Training_Data'
#target_dir = 'ISIC-2017_Training_Part1_GroundTruth/ISIC-2017_Training_Part1_GroundTruth'

class Dataloader:
    def __init__(self, img_folder, mask_folder, IMG_WIDTH=256, IMG_HEIGHT=256, split=0.8):
            self.IMG_WIDTH = IMG_WIDTH
            self.IMG_HEIGHT = IMG_HEIGHT
            self.img_folder = img_folder
            self.mask_folder = mask_folder
            self.split = split

    def load_image_path(self):
        # We get a list of paths to every image and mask
        input_img_paths = sorted(
            [
                os.path.join(self.img_folder, fname)
                for fname in os.listdir(self.img_folder)
                if fname.endswith(".jpg")
            ]
        )
        target_img_paths = sorted(
            [
                os.path.join(self.mask_folder, fname)
                for fname in os.listdir(self.mask_folder)
                if fname.endswith(".png")
            ]
        )
        return input_img_paths, target_img_paths

    def process_image(self, img_paths, mask_paths):
        X = np.zeros((len(img_paths), self.IMG_WIDTH, self.IMG_HEIGHT, 3), dtype=np.float32)
        Y = np.zeros((len(mask_paths), self.IMG_WIDTH, self.IMG_HEIGHT, 1), dtype=np.float32)

        # Normalize the data and turn them into arrays
        for i in range(len(img_paths)):
            img = img_paths[i]
            img = load_img(img, color_mode="rgb", target_size=(self.IMG_WIDTH, self.IMG_HEIGHT)) 
            img = img_to_array(img)
            X[i] = img.astype('float32') / 255.0
            
            mask = mask_paths[i]
            mask = load_img(mask, color_mode="grayscale", target_size=(self.IMG_WIDTH, self.IMG_HEIGHT))
            mask = img_to_array(mask)
            Y[i] = mask.astype('float32') / 255
        return X,Y

    # Training / Test / Validation splits
    def split_data(self, img_paths, X, Y):
        train_test_split = int(len(img_paths)*self.split)
        X_train = X[:train_test_split]
        Y_train = Y[:train_test_split]
        X_test = X[train_test_split:]
        Y_test = Y[train_test_split:]
        return X_train, Y_train, X_test, Y_test

    def get_XY_split(self):
        img_paths, mask_paths = self.load_image_path()
        X,Y = self.process_image(img_paths, mask_paths)
        X_train, Y_train, X_test, Y_test = self.split_data(img_paths, X,Y)
        return X_train, Y_train, X_test, Y_test



