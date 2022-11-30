import os
import sys
from dataset import DataLoader
import numpy as np
import utils_lib
from predict import Predictor

"""
File in which the YOLOv5 training is executed for the ISIC dataset.
This can only be run after dataset.py has been run to 
download/preprocess/arrange the dataset, and a ISIC_dataset.yaml
config file has been made. see README for how to make the yaml.
This will only work when run from:
PatternFlow_LC/recognition/s4532810-YOLO-xspinella
directory.

Training stats are saved to runs/train/exp_

Testing stats are saved to runs/val/exp_
"""
def Download_Preprocess_Arrange():
    """
    Complete setup of ALL files, datasets, and directories.
    After running this function, the user can train YOLOv5
    on the ISIC dataset. 
    This function also runs a few rudimentary tests on some functions
    and displays results to terminal/ saves test images to misc_tests.
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
    print(f"Test conversion of mask to box specs: Should return '[0.54296875, 0.56640625, 0.6078125, 0.7828125]\n'\
        {utils_lib.Mask_To_Box(gnd_truth)}")
    # Verify that img class lookup function is working for an isolated case:
    print(f"Test melanoma lookup function. Should return '(1, 'ISIC_0000002')'\n \
        {utils_lib.Find_Class_From_CSV(gnd_truth, dataloader.train_truth_gold)}")
    # Verify that label creation function works
    label, img_id = utils_lib.Get_YOLO_Label(gnd_truth, dataloader.train_truth_gold)
    np.savetxt(f"misc_tests/{img_id}.txt", np.array([label]), fmt='%f')
    # Verify that draw function is working
    utils_lib.Draw_Box(gnd_truth, label, "misc_tests/box_test_truth.png")
    utils_lib.Draw_Box(train_img, label, "misc_tests/box_test_img.png")

    ### generate a txt file for each img which specifies bounding box, and class of object ###
    # note that -> 0:melanoma, 1:!melanoma
    dataloader.Create_YOLO_Labels() 

    # Verify that box draw function from txt file label works
    label_fp = "yolov5_LC/data/labels/training/ISIC_0000002.txt"
    utils_lib.Draw_Box_From_Label(label_fp, train_img, "misc_tests/box_from_label.png")  

    ### Copy images to directories as required by yolov5 ###
    dataloader.Copy_Images()

    ### copy yaml file into correct YOLOv5 directory ###
    dataloader.Copy_Configs()

    print("Data setup complete.")

def train():
    """
    Executes the shell command for the training process
    using all the specified parameters, and the hyp.ISIC_2.yaml
    hyperparameter config file. Automatically applied the 
    augmentations specified in utils_2/albumentations.py
    """
    ### collect param info from user ###
    num_epochs = input("Please enter desired number of epochs (~450 is good): ")
    yolo_model = input("Please enter desired model (n/s/m/l/x): ")
    if yolo_model not in ['n', 's', 'm', 'l', 'x']:
        print("Invalid model")
        return 0
    
    ### Run training ###
    os.system(f"python3 yolov5_LC/train.py --img 640 --batch -1 --epochs {num_epochs} --data ISIC_dataset.yaml \
        --weights yolov5{yolo_model}.pt --hyp yolov5_LC/data/hyps/hyp.ISIC_2.yaml")

def test():
    """
    Runs the trained YOLOv5 model on the test dataset, and saves results to
    runs/val/exp_
    I've already run training and reccommend the weights at this path:
        /home/medicalrobotics/PatternFlow_LC/recognition/s4532810-YOLO-xspinella/v5m_exp4/v5m_exp4_train/weights/best.pt
    """
    ### Specify path to trained weights ###
    weights_path = input("Please paste the path to the YOLO weights to use. See train.py->test() for reccomended path: ")

    ### Test on test set ###
    # compute P, R, mAP
    os.system(f"python3 yolov5_LC/val.py --weights {weights_path} --data yolov5_LC/data/ISIC_test.yaml --img 640 --task test")
    # specify image and label paths
    test_dir_fp = "yolov5_LC/data/images/testing"
    test_labels_dir = "yolov5_LC/data/labels/testing"
    # specify paths to save distribution bar graphs at
    valid_bar_fp = "test_out/valid_bar"
    invalid_bar_fp = "test_out/invalid_bar"
    # compute IOU, classification accuracy, distribution info:
    avg_iou, tot_boxes, class_acc, tot_classes, pred_types, invalid_types = Compute_Mean_IOU_acc(\
                                test_dir_fp, test_labels_dir,weights_path)
    print(f"Average IOU: {avg_iou}, Valid Boxes: {tot_boxes}\n\
            Classification Accuracy: {class_acc}, Valid Classifications: {tot_classes}")
    # Plot data distributions
    utils_lib.Bar_Preds(pred_types, valid_bar_fp)
    utils_lib.Bar_Invalids(invalid_types, invalid_bar_fp)

def Compute_Mean_IOU_acc(dataset_fp: str, labels_fp: str, 
                        weights_path: str):
    """
    Finds the model's average IOU and classification accuracy of the 
    detections in the given dataset.
    :param dataset_fp: Filepath to the directory containing images to
                        run IOU analysis on.
    :param labels_fp: Filepath to the directory containing corresponding
                        labels
    :param weights_path: path to the model to evaluate
    :return: avg IOU, classification accuracy
    """
    # Instantiate and load trained model
    predictor = Predictor(weights_path, cpu=False)  # specify cpu=True to run on cpu
    
    ### Define vars for data to track (for plotting distributions) ###
    # box prediction data tracking
    box_preds = 0
    total_iou = 0
    # classification prediction data tracking
    total_class_preds = 0
    total_correct_preds = 0
    # valid prediction type data tracking
    num_TP = 0
    num_TN = 0
    num_FP = 0
    num_FN = 0
    # invalid detection data tracking
    P_badbox = 0
    N_badbox = 0
    P_failT = 0
    N_failT = 0
    
    ### iterate through dataset and find iou/acc ###
    images = os.listdir(dataset_fp)
    for image in images:
        # Find corresponding label
        img_fp = os.path.join(dataset_fp, image)
        img_id = utils_lib.Get_Img_ID(img_fp)
        label_fp = os.path.join(labels_fp, f"{img_id}.txt")
        # run model on image
        results = predictor.Predict_Img(img_fp)
        # IOU:
        iou = utils_lib.Compute_IOU(label_fp, results)
        if iou >= 0: 
            # if box is drawn, add to iou total
            box_preds += 1
            total_iou += iou
        # update class prediction info
        correct_pred, pred_type = utils_lib.Evaluate_Prediction(label_fp, results)
        # update distribution information
        if pred_type == 'TP':
            num_TP += 1
        elif pred_type == 'TN':
            num_TN += 1
        elif pred_type == 'FP':
            num_FP += 1
        elif pred_type == 'FN':
            num_FN += 1
        elif pred_type == 'P_badbox':
            P_badbox += 1
        elif pred_type == 'N_badbox':
            N_badbox += 1
        elif pred_type == 'P_failT':
            P_failT += 1
        elif pred_type == 'N_failT':
            N_failT += 1
        if correct_pred >= 0: 
            # if box valid/above iou threshold, iterate correct predictions
            total_class_preds += 1
            total_correct_preds += correct_pred
    # Compute avg IOU and class accuracy metrics
    avg_iou = total_iou/box_preds
    class_acc = total_correct_preds/total_class_preds
    # store distribution info
    pred_types = [num_TP, num_TN, num_FP, num_FN]
    invalid_types = [P_badbox, N_badbox, P_failT, N_failT]
    return [avg_iou, box_preds, class_acc, total_class_preds, pred_types, invalid_types]
        

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