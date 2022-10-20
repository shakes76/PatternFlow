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

def loadFile(dir, batch=16, size=(64, 64)):
    """Loads images in a directory (and its sub-directory)

    Args:
        dir: The target directory
        batch (int, optional): Batch size. Defaults to 16.
        size (tuple, optional): Image size. Defaults to (64, 64).

    Returns:
        6 datasets cooresponds to: trainingAD, trainingNC, validationAD, validationNC, testAD, testNC
    """
    print('>> Begin data loading')
    train_ad_dir = os.path.join(dir, 'train/AD')
    train_nc_dir = os.path.join(dir, 'train/NC')
    test_ad_dir = os.path.join(dir, 'test/AD')
    test_nc_dir = os.path.join(dir, 'test/NC')
    print('-Directory of the Training AD files is: {}'.format(train_ad_dir))
    print('-Directory of the Training NC files is: {}'.format(train_nc_dir))
    print('-Directory of the Testing AD files is: {}'.format(test_ad_dir))
    print('-Directory of the Testing NC files is: {}'.format(test_nc_dir))
    print('\n> 1/3 Loading Training Data...')
    train_ad_ds = utils.image_dataset_from_directory(train_ad_dir, 
                                                     labels = None,
                                                     label_mode = None,
                                                     validation_split=0.2,
                                                     subset="training",
                                                     seed=1,
                                                     image_size=size,
                                                     shuffle=True,
                                                     batch_size=batch)
    
    train_nc_ds = utils.image_dataset_from_directory(train_nc_dir, 
                                                     labels = None,
                                                     label_mode = None,
                                                     validation_split=0.2,
                                                     subset="training",
                                                     seed=1,
                                                     image_size=size,
                                                     shuffle=True,
                                                     batch_size=batch)
    print('\n> 2/3 Loading Validation Data...')
    valid_ad_ds = utils.image_dataset_from_directory(train_ad_dir, 
                                                     labels = None,
                                                     label_mode = None,
                                                     validation_split=0.2,
                                                     subset="validation",
                                                     seed=1,
                                                     image_size=size,
                                                     shuffle=True,
                                                     batch_size=batch)
    valid_nc_ds = utils.image_dataset_from_directory(train_nc_dir, 
                                                     labels = None,
                                                     label_mode = None,
                                                     validation_split=0.2,
                                                     subset="validation",
                                                     seed=1,
                                                     image_size=size,
                                                     shuffle=True,
                                                     batch_size=batch)
    print('\n> 3/3 Loading Test Data...')
    test_ad_ds = utils.image_dataset_from_directory(test_ad_dir, 
                                                     labels = None,
                                                     label_mode = None,
                                                     image_size=size,
                                                     shuffle=True,
                                                     batch_size=batch)
    
    test_nc_ds = utils.image_dataset_from_directory(test_nc_dir, 
                                                     labels = None,
                                                     label_mode = None,
                                                     image_size=size,
                                                     shuffle=True,
                                                     batch_size=batch)
    print('\n> Mapping datasets to greyscale...')
    train_ad_ds = train_ad_ds.map(transform_images)
    train_nc_ds = train_nc_ds.map(transform_images)
    valid_ad_ds = valid_ad_ds.map(transform_images)
    valid_nc_ds = valid_nc_ds.map(transform_images)
    test_ad_ds = test_ad_ds.map(transform_images)
    test_nc_ds = test_nc_ds.map(transform_images)
    
    print('\n>> Data loading complete')
    return train_ad_ds, train_nc_ds, valid_ad_ds, valid_nc_ds, test_ad_ds, test_nc_ds

def main():
    # Code for testing the functions
    tr_a, tr_n, v_a, v_n, te_a, te_n = loadFile('F:/AI/COMP3710/data/AD_NC/')

if __name__ == "__main__":
    main()

