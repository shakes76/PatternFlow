import torch
from skimage import io
from isics_data_setup import *

class YOLO():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.build_model()

    def build_model(self):
        # Medium size model is used given the simplicity of dataset
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5m')

    def train(self):
        self.print_info()
        self.model.train()

    def predict(self):
        pass
    
    def display_sample(self):
        image = io.imread("sample_image.jpg")
        results = self.model(image)
        results.show()

    def print_info(self):
        print(f"PyTorch Version: {torch.__version__}")