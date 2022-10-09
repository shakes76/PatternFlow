import os
import sys
from dataset import DataLoader
import numpy as np
import utils

"""
File in which the YOLOv5 training is executed for the ISIC dataset.
This can only be run after dataset.py has been run to 
download/preprocess/arrange the dataset, and a ISIC_dataset.yaml
config file has been made. see README for how to make the yaml.
This will only work when run from:
PatternFlow_LC/recognition/s4532810-YOLO-xspinella
directory.

Training stats are saved to runs/train/exp_

Testing states are saved to runs/val/exp_
"""
def Download_Preprocess_Arrange():
    """
    Complete setup of ALL files, datasets, and directories.
    After running this function, the user can train YOLOv5
    on the ISIC dataset. 
    This function also runs a few rudimentary tests on some functions
    and displays results to terminal.
    """
    ### Initialise dataloader, this implicitly: 
    #   deletes unwanted files, 
    #   downloads/extracts/resizes datasets 
    #   creates directory structure ###
    dataloader = DataLoader()

    ### Verify helper functions are working ###
    gnd_truth = dataloader.train_truth_PNG_ex + "/ISIC-2017_Training_Part1_GroundTruth/ISIC_0000002_segmentation.png"
    train_img = dataloader.train_data_ex + "/ISIC-2017_Training_Data/ISIC_0000002.jpg"
    # Verify that the bounding box code is working for an isolated case:
    print(f"Test conversion of mask to box specs: Should return '[0.54296875, 0.56640625, 0.6078125, 0.7828125]'\n\
        {dataloader.Mask_To_Box(gnd_truth)}")
    # Verify that img class lookup function is working for an isolated case:
    print(f"Test melanoma lookup function. Should return '(1, 'ISIC_0000002')'\n \
        {dataloader.Find_Class_From_CSV(gnd_truth, dataloader.train_truth_gold)}")
    # Verify that label creation function works
    label, img_id = dataloader.Get_YOLO_Label(gnd_truth, dataloader.train_truth_gold)
    np.savetxt(f"misc_tests/{img_id}.txt", np.array([label]), fmt='%f')
    # Verify that draw function is working
    utils.Draw_Box(gnd_truth, label, "misc_tests/box_test_truth.png")
    utils.Draw_Box(train_img, label, "misc_tests/box_test_img.png")

    ### generate a txt file for each img which specifies bounding box, and class of object ###
    # note that -> 0:melanoma, 1:!melanoma
    dataloader.Create_YOLO_Labels() 

    # Verify that box draw function from txt file label works
    label_fp = "yolov5_LC/data/labels/training/ISIC_0000002.txt"
    utils.Draw_Box_From_Label(label_fp, train_img, "misc_tests/box_from_label.png")  

    ### Copy images to directories as required by yolov5 ###
    dataloader.Copy_Images()

    ### copy yaml file into correct YOLOv5 directory ###
    dataloader.Copy_Configs()

    print("Data setup complete.")

def train():
    """
    Simply executes the shell command for the training process
    using all the proper parameters
    """
    ### Run training ###
    num_epochs = input("Please enter desired number of epochs (~400 is good): ")
    os.system(f"python3 yolov5_LC/train.py --img 640 --batch 16 --epochs {num_epochs} --data ISIC_dataset.yaml --weights yolov5s.pt")

def test():
    ### Test on test set ###
    # reccomended weights: yolov5_LC/runs/train/exp2/weights/best.pt
    weights_path = input("Please paste the path to the YOLO weights to use. See train.py for reccomended path: ")
    os.system(f"python3 yolov5_LC/val.py --weights {weights_path} --data yolov5_LC/data/ISIC_test.yaml --img 640")

if __name__ == "__main__":
    mode = int(input("Please enter desired mode (0 : download/preprocess/setup data/directories, 1 : train and test, 2 : train, 3 : test): "))
    if mode == 0:
        Download_Preprocess_Arrange()
    elif mode == 1:
        train()
        test()
    elif mode == 2:
        train()
    elif mode == 3:
        test()
    else:
        print("Invalid mode entered.")

