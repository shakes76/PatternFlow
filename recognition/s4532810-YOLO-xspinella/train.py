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
    print(f"Test conversion of mask to box specs: Should return '[0.54296875, 0.56640625, 0.6078125, 0.7828125]'\
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
    Simply executes the shell command for the training process
    using all the proper parameters
    """
    ### Run training ###
    num_epochs = input("Please enter desired number of epochs (~350 is good): ")
    os.system(f"python3 yolov5_LC/train.py --img 640 --batch -1 --epochs {num_epochs} --data ISIC_dataset.yaml --weights yolov5m.pt")

def test():
    """
    Runs the trained YOLOv5 model on the test dataset, and saves results to
    runs/val/exp_
    I've already run training and reccommend the weights at this path:
            yolov5_LC/runs/train/exp2/weights/best.pt    
    """
    ### Test on test set ###
    weights_path = input("Please paste the path to the YOLO weights to use. See train.py->test() for reccomended path: ")
    # Find P, R, mAP of dataset:
    os.system(f"python3 yolov5_LC/val.py --weights {weights_path} --data yolov5_LC/data/ISIC_test.yaml --img 640 --task test")
    # compute IOU, classification accuracy:
    test_dir_fp = "yolov5_LC/data/images/testing"
    test_labels_dir = "yolov5_LC/data/labels/testing"
    avg_iou, tot_boxes, class_acc, tot_classes = Compute_Mean_IOU_acc(\
                                test_dir_fp, test_labels_dir,weights_path)
    print(f"Average IOU: {avg_iou}, Valid Boxes: {tot_boxes}\n\
    Classification Accuracy: {class_acc}, Valid Classifications: {tot_classes}")

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
    predictor = Predictor()
    model = predictor.Load_Model(weights_path)
    
    ### iterate through dataset and find iou/acc ###
    images = os.listdir(dataset_fp)
    box_preds = 0
    total_iou = 0

    total_class_preds = 0
    total_correct_preds = 0
    for image in images:
        # Find corresponding label
        img_fp = os.path.join(dataset_fp, image)
        img_id = utils_lib.Get_Img_ID(img_fp)
        label_fp = os.path.join(labels_fp, f"{img_id}.txt")
        # run model on image
        results = predictor.Predict_Img(img_fp, model)
        # add to iou total
        iou = utils_lib.Compute_IOU(label_fp, results)
        if iou >= 0: # if valid iou
            box_preds += 1
            total_iou += iou
        # update class prediction info
        class_pred = utils_lib.Evaluate_Prediction(label_fp, results)
        if class_pred >= 0: # if valid/above iou threshold
            total_class_preds += 1
            total_correct_preds += class_pred
    avg_iou = total_iou/box_preds
    class_acc = total_correct_preds/total_class_preds
    return [avg_iou, box_preds, class_acc, total_class_preds]
        

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

# TODO:
# 1. how to find IOU graph
#   - IOU = overlap area / [(pred area + labelled area) - overlap area]
#   - make predict.py output IOU
#   - get IOU on test set
# 2. what does mAP actully mean
#   - "When a model has high recall but low precision, then the model 
#   - classifies most of the positive samples correctly but it has many 
#   - false positives(i.e. classifies many Negative samples as Positive).“

#   - “When a model has high precision but low recall, then the model is 
#   - accurate when it classifies a sample as Positive but it may classify 
#   - only some of the positive samples.”
# https://www.v7labs.com/blog/mean-average-precision
#   - 50
#   - 50-95
# 4. "suitable accuracy for classifiction" what is this based on
# 5. run training with a better yolo model
# 6. explanation of the following:
#   - box_loss: represents how well the algorithm can locate the centre of an 
#               object and how well the predicted bounding box covers an object. 
#   - obj_loss: Objectness is essentially a measure of the probability that an 
#               object exists in a proposed region of interest. If the objectivity 
#               is high, this means that the image window is likely to contain an object.
#   - cls_loss: class loss, gives an idea of how well the algorithm can predict 
#               the correct class of a given object.
#   - Precision(P): When the model predicts, how often does it predict correctly (does this include accounting for object classification?)
#   - Recall(R): has your model predicted every time that it should have predicted?
