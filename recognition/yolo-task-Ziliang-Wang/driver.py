from PIL import Image
from yolo import *

weights_path = 'results/epoch9, training loss3.97884,test_loss3.41089.pth'

anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
input_shape = [416, 416]

class_names = ['Lesion']
num_classes = 1

anchors = torch.Tensor([[10., 13.],
           [16., 30.],
           [33., 23.],
           [30., 61.],
           [62., 45.],
           [59., 119.],
           [116., 90.],
           [156., 198.],
           [373., 326.]])
num_anchors = 9
confidence = 0.5
nms_iou = 0.1
letterbox_image = False


def get_variable():
    """
    return variable from the driver.py
    """
    return weights_path, anchors_mask, input_shape, class_names, num_classes, anchors, num_anchors, confidence, nms_iou, letterbox_image


if __name__ == "__main__":
    yolo = YoloDetect()
    img_root = "img/ISIC_0016028.jpg"

    image = Image.open(img_root)

    image = yolo.detect_image(image)

    image.show()
