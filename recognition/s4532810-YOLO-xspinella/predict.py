from ast import Load
from curses import flash
from imp import load_module
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
    YOLOv5 model.
    """
    def __init__(self, weight_path: str, cpu=False):
        """
        Initialiser for the Predictor class - sets weight path
        and cpu usage flag. Also loads the model for this instance.
        :param weight_path: filepath to the yolov5 weight file to use.
        :param cpu: Loads on gpu if false, cpu if true.
        """
        self.weight_path = weight_path
        self.cpu = cpu
        self.model = self.Load_Model()

    def Load_Model(self):
        """
        Loads the weights at the specified path, and returns
        the loaded model. Uses the cpu to choose load device
        """
        if self.cpu:
            # Load to GPU
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.weight_path, device=torch.device('cpu'))
        else:
            # Load to GPU
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.weight_path)
        return model
    
    def Predict_Img(self, img_fp: str):
        """
        Uses the given model to predict bounding boxes/classification
        of the given image.
        :param img_fp: filepath to the image of interest
        :return: the yolov5-formatted results
        """
        results = self.model(img_fp)
        return results

    def Visualise_Comparison(self, img_fp: str, label_fp: str, results, out_fp: str):
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

    def Visualise_Prediction(self, img_fp: str, results, out_fp: str):
        """
        Runs the model on the given image, and saves a figure
        which visualises the detection.
        :param img_fp: filepath to the image of interest
        :param results: The prediction data.
        :param out_fp: filepath to save the output image as.
        """
        # Retrieve prediction image
        results.render()
        pred_img = results.ims[0]
        # plot using imshow
        plt.imshow(pred_img)
        plt.title("Predicted Label")
        plt.axis('off')

        plt.savefig(out_fp, bbox_inches='tight')
        
def Predictor_Example_Use():
    """
    Function used to show example use of the Predictor class
    """    
    ### Inference on Deployment ###
    # Load model - change filepath as desired
    weight_path = "/home/medicalrobotics/PatternFlow_LC/recognition/s4532810-YOLO-xspinella/v5m_exp2/v5m_exp2_train/weights/best.pt"
    predictor = Predictor(weight_path, cpu=False)

    # Define image to perform detection on
    img_fp = "yolov5_LC/data/images/testing/ISIC_0015184.jpg"
    out_fp = "pred_out/prediction_visual.png"
    # Perform detection
    results = predictor.Predict_Img(img_fp)
    # Visualise the detection
    predictor.Visualise_Prediction(img_fp, results, out_fp)
    # Display box specs (x1, y1, x2, y2), and classification
    print(results.pandas().xyxy[0].values.tolist()[0][:4])

    ### Labelled Set Comparisons ###
    # Run the model to retrieve results
    img_fp = "yolov5_LC/data/images/testing/ISIC_0015270.jpg"
    label_fp = "yolov5_LC/data/labels/testing/ISIC_0015270.txt"
    out_fp = "pred_out/prediction_comparison.png"
    results = predictor.Predict_Img(img_fp)

    # Visualise comparison between labelled and predicted
    predictor.Visualise_Comparison(img_fp, label_fp, results, out_fp)

    # Calculate IOU
    iou = utils_lib.Compute_IOU(label_fp, results)
    print(f"IOU: {iou}")

    # Calculate classification accuracies
    correct, type = utils_lib.Evaluate_Prediction(label_fp, results)
    print(f"correct prediction? {correct, type}")

if __name__ == "__main__":
    Predictor_Example_Use()