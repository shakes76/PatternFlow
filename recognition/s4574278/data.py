from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as transforms
from typing import List
from xml.etree import ElementTree as ET


class IsicDataSet(Dataset):
    def __init__(
        self, image_folder: str, annotation_folder: str, classes: List[str]
    ) -> None:
        super().__init__()
        self.image_folder = Path(image_folder)
        self.annotation_folder = Path(annotation_folder)
        self.classes = classes
        self._read_annotation(self.annotation_folder)
        self.images = [
            filename
            for filename in self.image_folder.iterdir()
            if filename.suffix.lower() == ".jpg"
            and self.annotations.__contains__(filename.name)
        ]
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        len(self.images)

    def __getitem__(self, index):
        image = self.transform(Image.open(self.images[index]).convert("RGB"))
        annotation = self.annotations[self.images[index].name]
        return image, annotation

    def _read_annotation(self, annotation_folder):
        self.annotations = {}
        for file in annotation_folder.iterdir():
            xml = ET.parse(file).getroot()
            image_name = xml.findtext("filename")
            annotation = []
            for object in xml.iter("object"):
                classname = object.find("name").text
                if classname not in self.classes:
                    continue
                class_index = self.classes.index(classname)
                bndbox = object.find("bndbox")
                bbox = [
                    int(float(bndbox.find("xmin").text)),
                    int(float(bndbox.find("ymin").text)),
                    int(float(bndbox.find("xmax").text)),
                    int(float(bndbox.find("ymax").text)),
                    class_index,
                ]
                annotation.append(bbox)
            self.annotations[image_name] = annotation


##########################################################
# Transforms
##########################################################


def resize(image_path, target_size, boxes, dtype=torch.float16):
    """resize image and annotation box"""
    image = Image.open(image_path)
    # Size of image
    image_height, image_width = image.size
    width, height = target_size
    # resize
    scale = min(width / image_width, height / image_height)
    new_width = int(image_width * scale)
    new_height = int(image_height * scale)
    delta_x = (width - new_width) // 2
    delta_y = (height - new_height) // 2

    # fill black background
    image = image.resize((new_width, new_height))
    new_image = Image.new("RGB", (width, height), (0, 0, 0))
    new_image.paste(image, (delta_x, delta_y))
    image_tensor = torch.tensor(new_image, dtype)

    # Adjust BBox accordingly
    if len(boxes) > 0:
        # new x min and x max
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * new_width / image_width + delta_x
        # new y min and y max
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * new_height / image_height + delta_y
        # x_min y_min must >= 0
        boxes[:, 0:2][boxes[:, 0:2] < 0] = 0
        # adjust the x_max, y_max according to the width and height
        boxes[:, 2][boxes[:, 2] > width] = width
        boxes[:, 3][boxes[:, 3] > height] = height
        # drop the boxes if any dimension collapse to zero
        box_w = boxes[:, 2] - boxes[:, 0]
        box_h = boxes[:, 3] - boxes[:, 1]
        boxes = boxes[torch.logical_and(box_w > 1, box_h > 1)]

    return image_tensor, boxes


if __name__ == "__main__":
    data = IsicDataSet("./dataset/input", "./dataset/annotation", ["lesion"])

    for i in range(1, 2):
        print(f"image: {data[i][0]} annotation: {data[i][1]}")
