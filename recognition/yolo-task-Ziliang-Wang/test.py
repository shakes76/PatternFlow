from PIL import Image
import torch
import yolo
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from yolo_utiles.plot import Plot_loss


# for load weights
weights_path = "results/weights/238epoch, training loss2.41873,test_loss2.26064.pth"

# for detect lesion
img_name = "ISIC_0016037.jpg"
img_root = "dataset/JPEGImages/" + img_name
xml_root = 'dataset/Annotations/'
JPEG_root = 'dataset/JPEGImages/'
weight_folder_path = r"results/weights/"






num_anchors = 9
confidence = 0.5
nms_iou = 0.1
letterbox_image = False

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

train_annotation = 'train.txt'
test_annotation = 'test.txt'
test_img_name = 'test_image_name.txt'

with open(train_annotation, encoding="utf-8") as f:
	train_lines = f.readlines()
with open(test_annotation, encoding="utf-8") as f:
	test_lines = f.readlines()
with open(test_img_name, encoding="utf-8") as f:
	test_img_name_lines = f.readlines()


def get_variable():
	"""
	return variable from driver.py
	"""
	return weights_path, anchors_mask, input_shape, class_names, num_classes, anchors, num_anchors, confidence, nms_iou, letterbox_image


def get_test_iou(weight_folder_path, yolo):
	"""
	get the test set iou with weights.
	"""
	weights_iou = []

	weights = os.listdir(weight_folder_path)
	sorted(weights, key=lambda i: i[:2])
	print(weights)
	for weight in weights:
		global weights_path
		epoch_avg_iou = []
		weights_path = weight_folder_path + weight
		yolov3 = yolo.YoloDetect(weights_path)

		for i, line in enumerate(test_img_name_lines):
			image = Image.open(JPEG_root + line[:-1])
			img_name = test_img_name_lines[i]
			iou = yolov3.detect_image(image, line, img_name, detect_image=False)
			epoch_avg_iou.append(iou)
		print(torch.mean(
			torch.tensor(epoch_avg_iou, dtype=torch.float,
						 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))))
		print(epoch_avg_iou)
		weights_iou.append(torch.mean(
			torch.tensor(epoch_avg_iou, dtype=torch.float,
						 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))).detach().cpu())
		Plot_loss(0, 0, weights_iou, epoch=len(weights)).plot_iou()


if __name__ == "__main__":
	yolo = yolo.YoloDetect(weights_path)
	image = Image.open(img_root)
	print(yolo.detect_image(image, img_root, img_name))

# check  the IOU with the weights
# print(get_test_iou(weight_folder_path, yolo))
