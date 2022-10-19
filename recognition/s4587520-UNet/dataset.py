#Dataset Loader to get images and segmentations from paths
import os
from torch.utils.data import Dataset
import matplotlib.image as mpimg

train_image_path = "./drive/MyDrive/ISIC Data/Train_Images/"
train_segmentation_path = "./drive/MyDrive/ISIC Data/Train_Segmentations/"


class ISIC_Dataset(Dataset):
  def __init__(self, image_path, segmentation_path):
    self.img_paths = []
    self.seg_paths = []

    for path in sorted(os.listdir(image_path)):
      if (path.endswith(".jpg")):
        self.img_paths.append(image_path + path)

    for path in sorted(os.listdir(segmentation_path)):
      if (path.endswith(".png")):
        self.seg_paths.append(segmentation_path + path)

  def __len__(self):
    return min(len(self.img_paths), len(self.seg_paths))

  def __getitem__(self, id):
    image = mpimg.imread(self.img_paths[id])
    mask = mpimg.imread(self.seg_paths[id])
    return (image, mask)
