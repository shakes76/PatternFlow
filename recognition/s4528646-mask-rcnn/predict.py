import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from dataset import ISICDataset, get_transform
from modules import get_model
import cv2
import torch
from torchvision.ops import nms

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def mask_thresh(thresh=0.5):
    pass

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou


val_data = ISICDataset(
    image_folder_path="./data/ISIC-2017_Training_Data", 
    mask_folder_path="./data/ISIC-2017_Training_Part1_GroundTruth", 
    diagnoses_path="./data/ISIC-2017_Training_Part3_GroundTruth.csv",
    device=device,
    transform=get_transform(True)
    )
val_dataloader = torch.utils.data.DataLoader(
    val_data, 
    batch_size=1, 
    shuffle=True, 
    collate_fn=lambda x:list(zip(*x))
    )

model = get_model()
model.float()
model.load_state_dict(torch.load("./Mask_RCNN_ISIC.pt"))
model.eval()
image, targets = val_data[6]
predictions = model([image])
image = np.array(image.detach().cpu())
image = np.swapaxes(np.swapaxes(image, 0, 2), 1, 0)
fig, ax = plt.subplots()
ax.imshow(image[...,::-1])
boxes = predictions[0]["boxes"]
scores = predictions[0]["scores"]
for i in range(boxes.shape[0]):
    bbox = boxes[i].detach()
    rect = Rectangle((bbox[0], bbox[1]), bbox[3] - bbox[1], bbox[2] - bbox[0], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    target_bbox = targets["boxes"][0]
    rect = Rectangle((target_bbox[0], target_bbox[1]), target_bbox[2] - target_bbox[0], target_bbox[3] - target_bbox[1], linewidth=1, edgecolor='b', facecolor='none')
    ax.add_patch(rect)
    print(bb_intersection_over_union(bbox, target_bbox))
    break