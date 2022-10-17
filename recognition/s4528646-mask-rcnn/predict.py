import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from dataset import ISICDataset, get_transform
from modules import get_model
import cv2
import torch
from torchvision.ops import nms, box_iou

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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
image, targets = val_data[5]
predictions = model([image])
image = np.array(image.detach().cpu())
image = image.transpose((1,2,0))
fig, ax = plt.subplots()
ax.imshow(image[...,::-1])
boxes = predictions[0]["boxes"]
scores = predictions[0]["scores"]
for i in range(boxes.shape[0]):
    bbox = boxes[i].detach()
    rect = Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    target_bbox = targets["boxes"][0]
    rect = Rectangle((target_bbox[0], target_bbox[1]), target_bbox[2] - target_bbox[0], target_bbox[3] - target_bbox[1], linewidth=1, edgecolor='b', facecolor='none')
    ax.add_patch(rect)
    print(box_iou(boxes, targets["boxes"]))
    break