import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms as T
import pandas as pd

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

def Get_Gnd_Truth_Img_ID(img_fp: str):
    """
    finds the image ID of the given gnd truth img mask.
    :param img_fp: the img to find the id of
    :return: the mask image id
    """   
    ### Find image id from the fp ###
    # remove directories from fp string
    last_slash_idx = img_fp.rfind('/')
    img_fp = img_fp[last_slash_idx+1:]
    # extract img id
    dot_idx = img_fp.rfind('_')
    img_id = img_fp[0:dot_idx]
    return img_id