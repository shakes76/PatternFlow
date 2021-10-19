import torch
import torchvision
from skimage import io
from isics_data_loader import *

class YOLO():
    CLASS_NAMES = ['BG', 'Lesion']

    def __init__(self, dir, batch_size, valid_split=0.2):
        self.dir = dir
        self.batch_size = batch_size
        self.get_and_split_data(valid_split)
        self.build_model()

    def build_model(self):
        # Medium size model given the simplicity of dataset
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5m')

    def train(self):
        self.print_info()
    
    def display_sample(self):
        image = io.imread("sample_image.jpg")
        results = self.model(image)
        results.show()

    def print_info(self):
        print(f"PyTorch Version: {torch.__version__}")
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'Dataset directory: {self.dir}')