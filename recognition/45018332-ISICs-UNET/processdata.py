import os
import random
import glob
import shutil

def move_files(flist, dest):
    for f in flist:
        shutil.move(f, dest)

def rearr_folders(data_path, img_path, mask_path):
    #new split folders for images
    data_path = './ISIC2018_Task1-2_Training_Data'
    img_path = '/ISIC2018_Task1-2_Training_Input_x2'
    mask_path = '/ISIC2018_Task1_Training_GroundTruth_x2'

    folders = ['/train_img', '/train_mask', '/test_img', '/test_mask', '/val_img', '/val_mask']
    for folder in folders:
        os.makedirs(data_path+folder)

    img_paths = sorted(glob.glob(data_path+img_path+'/*.jpg'))
    mask_paths = sorted(glob.glob(data_path+mask_path+'/*.png'))

    random.seed(42)
    random.shuffle(img_paths)

    random.seed(42)
    random.shuffle(mask_paths)

    #split data to train-test-val of ratio 7:1:2
    train_len = int(0.7*len(img_paths))
    test_len = int(0.8*len(img_paths))

    train_imgs = img_paths[:train_len]
    test_imgs = img_paths[train_len:test_len]
    val_imgs = img_paths[test_len:]

    train_masks = mask_paths[:train_len]
    test_masks = mask_paths[train_len:test_len]
    val_masks = mask_paths[test_len:]

    #move files to new folders
    move_files(train_imgs, (data_path+folders[0]))
    move_files(train_masks, (data_path+folders[1]))
    move_files(test_imgs, (data_path+folders[2]))
    move_files(test_masks, (data_path+folders[3]))
    move_files(val_imgs, (data_path+folders[4]))
    move_files(val_masks, (data_path+folders[5]))