from curses import flash
import sys
sys.path.append("yolov5_LC")
import os
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import utils_lib
from PIL import Image
import numpy as np

"""
File used to show example usage of the trained model
"""

class Predictor():
    """
    Class with functionality for operational usage of the trained
    YOLOv5 model
    """
    def __init__(self):
        """
        Initialiser for the Predictor class
        """

    def Load_Model(self, weight_path: str):
        """
        Loads the weights at the specified path, and returns
        the loaded model
        :param weight_path: the path to the desired YOLOv5 weights
        """
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=weight_path)  # local model
        return model
    
    def Predict_Img(self, img_fp: str, model):
        """
        Uses the given model to predict bounding boxes/classification
        of the given image.
        :param img_fp: filepath to the image of interest
        :param model: model to use to evaluate image
        :return: the yolov5-formatted results
        """
        results = model(img_fp)
        return results

    def Visualise_Prediction(self, img_fp: str, label_fp: str, results, out_fp: str):
        """
        Runs the model on the given image, and saves a figure
        which compares the prediction to the actual label.
        :param img_fp: filepath to the image of interest
        :param label_fp: filepath to corressponding img label
        :param results: The prediction data.
        :param out_fp: filepath to save the output image as.
        """
        # Retrieve prediction image
        results.render()
        pred_img = results.ims[0]
        # Retrieve labelled img
        utils_lib.Draw_Box_From_Label(label_fp, img_fp, out_fp)
        label_img = Image.open(out_fp)
        # Arrange subplot for comparison
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(label_img)
        ax1.set(title="Labelled Image")
        ax1.axis('off')

        ax2.imshow(pred_img)
        ax2.set(title="Predicted Label")
        ax2.axis('off')

        fig.set_size_inches(14, 7)
        plt.savefig(out_fp, bbox_inches='tight')
        


def Predictor_Example_Use():
    """
    Function used to show example use of the Predictor class
    """    
    # Load model
    predictor = Predictor()
    model = predictor.Load_Model("yolov5_LC/runs/train/exp2/weights/best.pt")

    # Predict
    img_fp = "yolov5_LC/data/images/testing/ISIC_0015184.jpg"
    label_fp = "yolov5_LC/data/labels/testing/ISIC_0015184.txt"
    out_fp = "misc_tests/prediction_comparison.png"
    results = predictor.Predict_Img(img_fp, model)

    # Visualise
    predictor.Visualise_Prediction(img_fp, label_fp, results, out_fp)

    # Calculate IOU
    iou = utils_lib.Compute_IOU(label_fp, results)
    print(f"IOU: {iou}")

    # Calculate classification accuracies
    correct = utils_lib.Evaluate_Prediction(label_fp, results)
    print(f"correct prediction? {correct}")
    # Quantify results
    # print(results.pandas().xyxy[0].values.tolist()[0][:4])
  

if __name__ == "__main__":
    Predictor_Example_Use()