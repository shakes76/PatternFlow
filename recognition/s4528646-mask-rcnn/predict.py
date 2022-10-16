import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from modules import get_model
import cv2
from torchvision.ops import nms

def mask_thresh(thresh=0.5):
    pass

def intersection_over_union():
    pass


val_data = ISICDataset(
    image_folder_path="./data/ISIC-2017_Validation_Data", 
    mask_folder_path="./data/ISIC-2017_Validation_Part1_GroundTruth", 
    diagnoses_path="./data/ISIC-2017_Validation_Part3_GroundTruth.csv",
    device=device,
    )
val_dataloader = torch.utils.data.DataLoader(
    train_data, 
    batch_size=1, 
    shuffle=True, 
    collate_fn=lambda x:list(zip(*x))
    )

model = get_model()
model.double()
model.load_state_dict(torch.load("./Mask_RCNN_ISIC.torch"))
model.eval()
image, targets = val_data[56]
predictions = model([image])
image = np.array(image.detach().cpu())
image = np.swapaxes(image, 0, 2)
fig, ax = plt.subplots()
ax.imshow(image[...,::-1])
boxes = predictions[0]["boxes"]
scores = predictions[0]["scores"]
boxes_nms = nms(boxes, scores, 0.4)
bbox = boxes[boxes_nms[0]].detach()
rect = Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1, edgecolor='r', facecolor='none')
ax.add_patch(rect)