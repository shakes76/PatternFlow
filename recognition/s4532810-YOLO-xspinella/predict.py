import torch
import matplotlib.pyplot as plt
import utils_lib
from PIL import Image
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

    def Visualise_Prediction(self, img_fp: str, label_fp: str, model, out_fp: str):
        """
        Runs the model on the given image, and saves a figure
        which compares the prediction to the actual label.
        :param img_fp: filepath to the image of interest
        :param label_fp: filepath to corressponding img label
        :param model: model to use to evaluate image
        :param out_fp: filepath to save the output image as.
        """
        # Retrieve prediction image
        results = self.Predict_Img(img_fp, model)
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
        # TODO: calculate IOU
        return results


def Predictor_Example_Use():
    """
    Function used to show example use of the Predictor class
    """    
    # Load model
    predictor = Predictor()
    model = predictor.Load_Model("yolov5_LC/runs/train/exp2/weights/best.pt")

    # Predict
    img_fp = "yolov5_LC/data/images/training/ISIC_0000002.jpg"
    label_fp = "yolov5_LC/data/labels/training/ISIC_0000002.txt"
    out_fp = "misc_tests/prediction_comparison.png"

    # Visualise
    results = predictor.Visualise_Prediction(img_fp, label_fp, model, out_fp)

    # Quantify results
    print(results.pandas().xyxy[0])

if __name__ == "__main__":
    Predictor_Example_Use()