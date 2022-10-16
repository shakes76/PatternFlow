# This file contains the data loader
import numpy as np
import os
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import image
from tensorflow.keras import utils
import matplotlib.pyplot as plt

def transform_images(img, label):
    # transform to [0,1]
    img = image.rgb_to_grayscale(img)
    img = img / 255
    return img, label

def loadFile(dir):
    print('>> Begin data loading')
    train_dir = os.path.join(dir, 'train')
    print('-Directory of the Training files is: {}'.format(train_dir))
    print('\n> 1/2 Loading Training Data...')
    train_ds = utils.image_dataset_from_directory(train_dir, 
                                                     labels = 'inferred', 
                                                     class_names =['AD', 'NC'],
                                                     validation_split=0.3,
                                                     subset="training",
                                                     seed=1,
                                                     image_size=(256, 240),
                                                     shuffle=True,
                                                     batch_size=8)
    print('\n> 2/2 Loading Validation Data...')
    valid_ds = utils.image_dataset_from_directory(train_dir, 
                                                     labels = 'inferred', 
                                                     class_names =['AD', 'NC'],
                                                     validation_split=0.3,
                                                     subset="validation",
                                                     seed=1,
                                                     image_size=(256, 240),
                                                     shuffle=True,
                                                     batch_size=8)
    print('\n> Mapping datasets to greyscale...')
    train_ds = train_ds.map(transform_images)
    valid_ds = valid_ds.map(transform_images)
    
    print('\n>> Data loading complete')
    return train_ds, valid_ds
    
def plotExample(ds):
    batch = ds.take(1)
    for img, label in batch:
        plt.axis("off")
        plt.imshow((img.numpy()*255).astype("int32")[0], cmap='gray', vmin=0, vmax=255)
        plt.show()
        break

def main():
    # Code for testing the functions
    t, v = loadFile('F:/AI/COMP3710/data/AD_NC/')
    plotExample(t)

if __name__ == "__main__":
    main()

