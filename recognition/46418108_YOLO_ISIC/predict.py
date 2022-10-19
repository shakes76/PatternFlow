import torch
import cv2
from PIL import Image

def load_model(path='best.pt'):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=True)
    return model

def predict(model, image_path):
    im = Image.open(image_path)
    results = model(im)
    results.save()

model = load_model()
predict(model, "path_to_image")
