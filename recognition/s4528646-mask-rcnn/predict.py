import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from dataset import ISICDataset, get_transform
from modules import get_model
import torch
from torchvision.ops import nms, box_iou

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

val_data = ISICDataset(
    image_folder_path="./data/ISIC-2017_Validation_Data", 
    mask_folder_path="./data/ISIC-2017_Validation_Part1_GroundTruth", 
    diagnoses_path="./data/ISIC-2017_Validation_Part3_GroundTruth.csv",
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
model.load_state_dict(torch.load("./Mask_RCNN_ISIC4.pt"))
model.eval()
image, targets = val_data[73]
predictions = model([image])
image = np.array(image.detach().cpu())
image = image.transpose((1,2,0))
fig, ax = plt.subplots()
ax.imshow(image[...,::-1])
boxes = predictions[0]["boxes"]
scores = predictions[0]["scores"]
iou = box_iou(boxes, targets["boxes"])
idx = torch.argmax(iou)
bbox = boxes[idx].detach()
mask = predictions[0]["masks"][idx]
rect = Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1, edgecolor='r', facecolor='none')
rx, ry = rect.get_xy()
cx = rx + rect.get_width()/8.0
cy = ry - rect.get_height()/22.0
label = "Melanoma" if targets["labels"][0] == 2 else "Non-Melanoma"
l = ax.annotate(
        label,
        (cx, cy),
        fontsize=7,
        # fontweight="bold",
        color="r",
        ha='center',
        va='center'
      )
ax.add_patch(rect)
# Classification
print("Expected:", targets["labels"].item(), "Predicted:", predictions[0]["labels"][idx].item())
# IoU
print("IoU:", iou[idx])
# Show mask prediction
plt.imshow(predictions[0]["masks"][idx][0].detach() > 0.5)