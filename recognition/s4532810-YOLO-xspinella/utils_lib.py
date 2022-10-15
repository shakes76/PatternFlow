import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms as T
import pandas as pd
import torch
from yolov5_LC.utils_2.metrics import box_iou, bbox_iou

"""
Function library for the patternflow YOLO project
"""

def Draw_Box_From_Label(label_fp: str, img_fp: str, out_fp: str):
    """
    Draws bounding box on either actual image,
    or gnd truth segmentation, using the YOLO-format label.
    :param label_fp: Filepath to the YOLO-format label text file.
    :param img_fp: Filepath to the img of interest.
    :param out_fp: output file filepath.
    """
    box_spec = list(np.loadtxt(label_fp))
    Draw_Box(img_fp, box_spec, out_fp)

def Draw_Box(img_fp: str, box_spec: list, out_fp: str):
    """
    Draws the specified box on the given image
    :param img_fp: filepath to the image
    :param box_spec: box size/location/class specification as:
                        [class, centre_x, centre_y, width, height]
    :param out_fp: output file location
    """
    ### open image with cv2, save image size, define box specs ###
    img = cv2.imread(img_fp)
    height, width, _ = img.shape
    class_id, c_x, c_y, w, h = box_spec
    class_id = int(class_id)

    ### define specifications for drawing box for different classes ###
    class_list = ['Not Melanoma', 'Melanoma']
    colours = [(0, 0, 255), (151, 157, 255)]
    label = class_list[class_id]
    colour = colours[class_id] 

    ### redefine box location in cv2 format, un-normalise co-ords ###
    # This box format appears to be: [mid of left edge, mid of top edge, width, height]
    box = [int((c_x - 0.5*w)* width), int((c_y - 0.5*h) * height), int(w*width), int(h*height)]
    cv2.rectangle(img, box, color=colour)   # bounding box
    cv2.rectangle(img, (box[0], box[1] - 20), (box[0] + box[2], box[1]), colour, -1)    # label box 
    cv2.putText(img, class_list[class_id], (box[0], box[1] - 5), cv2.FONT_HERSHEY_DUPLEX, .5, (255,255,255))
    cv2.imwrite(out_fp, img)

def Get_YOLO_Label(mask_path: str, csv_path: str):
    """
    :param mask_path: path to mask segmentation of image to produce
                        label for
    :param csv_path: path to csv which contains the melanoma classification
                        for this image id
    :return: The YOLO-format label for this image id:
                normalised([class, C_x, C_y, w, h])
                and the img id as a string
    """
    ### Find the box specs and class spec ###
    normalised_box_spec = Mask_To_Box(mask_path)
    # remember 0:!melanoma, 1:melanoma
    melanoma_class, img_id = Find_Class_From_CSV(mask_path, csv_path)

    ### Concatenate in correct order ###
    normalised_box_spec.insert(0, float(melanoma_class))
    return normalised_box_spec, img_id 

def Mask_To_Box(img_fp: str):
    """
    Converts given segment mask into bounding box specification:
    x, y, w, h
    :param img: filepath to segment mask of one of the lesions
    :return: Bounding box definition as [centre_x, centre_y, width, height]
    """
    # Open image and convert to array
    img = Image.open(img_fp)
    img_arr = np.array(img)

    # define vars for pointing out the bounds of the box
    min_left = np.inf
    max_right = -np.inf
    min_up = np.inf
    max_down = -np.inf
    # found the bounds of the box:
    for i in range(0, len(img_arr)):        # Rows
        for j in range(0, len(img_arr[0])): # Cols
            if img_arr[i][j] > 0:
                min_left = min(min_left, j)
                max_right = max(max_right, j)
                min_up = min(min_up, i)
                max_down = max(max_down, i)
    # redefine as centre_x, centre_y, width, height
    w = max_right - min_left
    h = max_down - min_up
    c_x = min_left + (w/2)
    c_y = min_up + (h/2)
    # bounding box params are normalised amd returned
    return [c_x/640, c_y/640, w/640, h/640]

def Find_Class_From_CSV(img_fp: str, csv_fp: str):
    """
    Find the class (0:!melanoma, 1:melanoma) of the given
    filename, by matching the id from the filename to
    the id in the row of the csv
    :param img_fp: filepath of the gnd truth image of interest
    :param csv_fp: filepath of corresponding csv file for classification
    :return: the classification of the object in this img and the img_id
    """
    ### Find image id from the fp ###
    img_id = Get_Gnd_Truth_Img_ID(img_fp)

    ### Find the classification from given csv ###
    # find row corresponding to the img_id 
    img_df = pd.read_csv(csv_fp)
    img_arr = img_df.values
    id_row = np.NaN
    i = 0
    for row in img_arr:
        if row[0] == img_id:
            id_row = i
            break
        i += 1
    if id_row == np.NaN:
        raise LookupError("The image ID was not found in CSV file")
    # return the melanoma classification (0:!melanoma, 1:melanoma)
    return int(img_arr[id_row][1]), img_id

def Get_Gnd_Truth_Img_ID(truth_img_fp: str):
    """
    finds the image ID of the given gnd truth img mask.
    :param img_fp: the img to find the id of
    :return: the mask image id
    """   
    ### Find image id from the fp ###
    # remove directories from fp string
    last_slash_idx = truth_img_fp.rfind('/')
    truth_img_fp = truth_img_fp[last_slash_idx+1:]
    # extract img id
    dot_idx = truth_img_fp.rfind('_')
    img_id = truth_img_fp[0:dot_idx]
    return img_id

def Get_Img_ID(img_fp: str):
    """
    finds the image ID of the given image filepath.
    :param img_fp: the img to find the id of
    :return: the image id
    """   
    ### Find image id from the fp ###
    # remove directories from fp string
    last_slash_idx = img_fp.rfind('/')
    img_fp = img_fp[last_slash_idx+1:]
    # extract img id
    dot_idx = img_fp.rfind('.')
    img_id = img_fp[0:dot_idx]
    return img_id

def Get_Box_From_Label(label_fp: str):
    """
    Extract the box dimensions from the given label
    :param label_fp: filepath to the label of interest
    :return: The box spec in Cx, Cy, w, h format
    """
    class_id, c_x, c_y, w, h = list(np.loadtxt(label_fp))
    return [c_x, c_y, w, h]

def Convert_Box_Format(box_spec: list):
    """
    Converts the box_spec in [Cx, Cy, w, h] format to
    [x1, y1, x2, y2] format
    :param box_spec: box spec in [Cx, Cy, w, h] format 
    :return: box spec in [x1, y1, x2, y2] format
    """
    Cx, Cy, w, h = box_spec
    w_, h_ = w/2, h/2
    b_x1, b_x2, b_y1, b_y2 = Cx - w_, Cx + w_, Cy - h_, Cy + h_
    return [b_x1, b_y1, b_x2, b_y2]

def Revert_Box_Format(box_spec: list):
    """
    Converts the box_spec in [x1, y1, x2, y2] format to
    [Cx, Cy, w, h] format
    :param box_spec: box spec in [x1, y1, x2, y2] format 
    :return: box spec in [Cx, Cy, w, h] format
    """
    x1, y1, x2, y2 = box_spec
    w, h = x2 - x1, y2 - y1
    Cx, Cy = w/2, h/2 
    return [Cx, Cy, w, h]

def Compute_IOU(label_fp: str, results):
    """
    Computes the Intersection over union of the predicted box
    in comparison to the label box.
    :param img_fp: filepath to the image of interest
    :param label_fp: filepath to the corresponding label.
    :param results: the results as returned by the model for this img.
    :return: The IOU for the predicted box
    """
    ### Retrieve label and prediction box specs in x1, y1, x2, y2 format ###
    label_box = Get_Box_From_Label(label_fp)
    label_box = Convert_Box_Format(label_box)

    ### verify there is actually a box ###
    num_detects = len(results.pandas().xyxy[0].index)
    if num_detects < 1:
        return -1   
    
    ### iterate through all boxes, calc iou ###
    iou_tot = 0
    i = 0
    while i < num_detects:
        pred_box = results.pandas().xyxy[0].values.tolist()[i][:4]
        # normalise:
        pred_box = (np.array(pred_box)/640).tolist() 

        ### Find IOU ###
        iou = bbox_iou(torch.FloatTensor([pred_box]), 
                        torch.FloatTensor([label_box]), xywh=False)
        iou_tot += iou.item()
        i += 1

    ### calc overall iou ###
    avg_iou = iou_tot/num_detects
    return avg_iou

def Evaluate_Prediction(label_fp: str, results):
    """
    Evaluates the correctness of the classification made
    by the model - only considers images when
    the model actually detects an object successfully:
    i.e. IOU >= 0.5, and the model doesn't predict more than 1 object
    (all unaugmented images only contain one lesion)
    :param label_fp: the label for the image of interest
    :param results: the results as returned by the model for this img.
    :return: 0 if incorrect, 1 if correct, -1 if no object detected
            second return term is -1  or cwhether the prediction was
            TN, TP, FP, FN or I:invalid
    """
    ### find actual class ###
    label_class = list(np.loadtxt(label_fp))[0]

    ### Check if valid detection ###
    if not (len(results.pandas().xyxy[0].index) == 1):
        # all (unaugmented) images contain only one lesion
        # So if the model predicts more than one object, 
        # we do not consider the classification,
        # if the length is 0, then no prediction was made,
        # so we also disregard this
        if label_class:
            return -1, 'P_badbox'    # positive label
        else:
            return -1, 'N_badbox'    # negative label
    if Compute_IOU(label_fp, results) < 0.5:    # Invalid IOU
        if label_class:
            return -1, 'P_failT' 
        else:
            return -1, 'N_failT'   

    ### Check predicted vs actual class ### 
    pred_class = results.pandas().xyxy[0].values.tolist()[0][5]
    if not (pred_class == label_class):     # incorrect prediciton
        if pred_class:
            return 0, 'FP'  # False Positive
        else:
            return 0, 'FN'  # False Negative

    if pred_class:    
        return 1, 'TP'      # True Positive
    else:
        return 1, 'TN'      # True Negative 

def Bar_Preds(pred_types: list, out_fp: str):
    """
    Plots a bar graph of TP, TN, FP, FN, Tot_P, Tot_N.
    :param pred_types: list in format:
                [num_TP, num_TN, num_FP, num_FN]
    :param out_fp: output file path
    """
    num_TP, num_TN, num_FP, num_FN = pred_types
    tot_p = num_TP + num_FN
    tot_n = num_FP + num_TN
    bar_list = [num_TP, num_TN, num_FP, num_FN, tot_p, tot_n]

    sections = ['TP', 'TN', 'FP', 'FN', 'Tot_P', 'Tot_N']

    fig, ax = plt.subplots()
    ax.clear()
    bars = ax.bar(sections, bar_list)

    ax.bar_label(bars)
    ax.set_xlabel("Valid Prediciton Type")
    ax.set_ylabel("Number of Predictions")

    plt.title("Analysis of Valid Predictions")
    plt.savefig(out_fp)

def Bar_Invalids(invalid_types: list, out_fp: str):
    """
    Plots a bar graph which summarises invalid 
    predictions (multiple boxes/no boxes) and (pos/neg)
    :param invalid_types: list in format:
                ['P_badbox', 'N_badbox', 'P_failT', 'N_failT']
    :param out_fp: output file path
    """
    P_badbox, N_badbox, P_failT, N_failT = invalid_types
    tot_p = P_badbox + P_failT
    tot_n = N_badbox + N_failT
    bar_list = [P_badbox, N_badbox, P_failT, N_failT, tot_p, tot_n]

    sections = ['P_badbox', 'N_badbox', 'P_failT',\
         'N_failT', 'Tot_P', 'Tot_N']

    fig, ax = plt.subplots()
    ax.clear()
    bars = ax.bar(sections, bar_list)

    ax.bar_label(bars)
    ax.set_xlabel("Invalid Detection Type")
    ax.set_ylabel("Number of Detections")

    plt.title("Analysis of Invalid Box Detections")
    plt.savefig(out_fp)