import torch
import cv2
from PIL import Image

def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
    return model


def predict(model, image_path='ISIC_0010251.jpg'):
    im = Image.open(image_path)
    results = model(im)
    results.show()

model = load_model()
predict(model)