#Dataset Loader to get images and segmentations from paths
import os
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image

train_image_path = "./ISIC-2017_Training_Data"
train_segmentation_path = "./ISIC-2017_Training_Part1_GroundTruth"

class ISIC_Dataset(Dataset):
  def __init__(self, image_path, segmentation_path):
    self.img_paths = []
    self.seg_paths = []
    self.transform = ToTensor()

    for path in sorted(os.listdir(image_path)):
      if (path.endswith(".jpg")):
        self.img_paths.append(image_path + path)

    for path in sorted(os.listdir(segmentation_path)):
      if (path.endswith(".png")):
        self.seg_paths.append(segmentation_path + path)

  def __len__(self):
    return min(len(self.img_paths), len(self.seg_paths))

  def __getitem__(self, id):
    image = self.transform(Image.open(self.img_paths[id]))
    mask = self.transform(Image.open(self.seg_paths[id]))
    return (image, mask)

train_dataset = ISIC_Dataset(train_image_path, train_segmentation_path)