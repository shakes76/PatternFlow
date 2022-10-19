# This file contains the data loader
import os
from tensorflow import image
from tensorflow.keras import utils

def transform_images(img):
    # transform to grayscale and standardize to [0,1]
    img = image.rgb_to_grayscale(img)
    img = img / 255.0
    # print(img)
    return img

def loadFile(dir, batch=16):
    print('>> Begin data loading')
    train_ad_dir = os.path.join(dir, 'train/AD')
    train_nc_dir = os.path.join(dir, 'train/NC')
    print('-Directory of the Training AD files is: {}'.format(train_ad_dir))
    print('-Directory of the Training NC files is: {}'.format(train_nc_dir))
    print('\n> 1/2 Loading Training Data...')
    train_ad_ds = utils.image_dataset_from_directory(train_ad_dir, 
                                                     labels = None,
                                                     label_mode = None,
                                                     validation_split=0.2,
                                                     subset="training",
                                                     seed=1,
                                                     image_size=(224, 224),
                                                     shuffle=True,
                                                     batch_size=batch)
    
    train_nc_ds = utils.image_dataset_from_directory(train_nc_dir, 
                                                     labels = None,
                                                     label_mode = None,
                                                     validation_split=0.2,
                                                     subset="training",
                                                     seed=1,
                                                     image_size=(224, 224),
                                                     shuffle=True,
                                                     batch_size=batch)
    print('\n> 2/2 Loading Validation Data...')
    valid_ad_ds = utils.image_dataset_from_directory(train_ad_dir, 
                                                     labels = None,
                                                     label_mode = None,
                                                     validation_split=0.2,
                                                     subset="validation",
                                                     seed=1,
                                                     image_size=(224, 224),
                                                     shuffle=True,
                                                     batch_size=batch)
    valid_nc_ds = utils.image_dataset_from_directory(train_nc_dir, 
                                                     labels = None,
                                                     label_mode = None,
                                                     validation_split=0.2,
                                                     subset="validation",
                                                     seed=1,
                                                     image_size=(224, 224),
                                                     shuffle=True,
                                                     batch_size=batch)
    print('\n> Mapping datasets to greyscale...')
    train_ad_ds = train_ad_ds.map(transform_images)
    train_nc_ds = train_nc_ds.map(transform_images)
    valid_ad_ds = valid_ad_ds.map(transform_images)
    valid_nc_ds = valid_nc_ds.map(transform_images)
    
    print('\n>> Data loading complete')
    return train_ad_ds, train_nc_ds, valid_ad_ds, valid_nc_ds

def main():
    # Code for testing the functions
    t_a, t_n, v_a, v_n = loadFile('F:/AI/COMP3710/data/AD_NC/')

if __name__ == "__main__":
    main()

