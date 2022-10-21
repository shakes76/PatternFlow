from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as transforms
from typing import List
from xml.etree import ElementTree as ET


class IsicDataSet(Dataset):
    def __init__(
        self,
        image_folder: str,
        annotation_folder: str,
        classes: List[str],
        image_shape=(512, 512),
    ) -> None:
        super().__init__()
        self.image_folder = Path(image_folder)
        self.annotation_folder = Path(annotation_folder)
        self.classes = classes
        self.image_shape = image_shape
        self._read_annotation(self.annotation_folder)
        self.images = [
            filename
            for filename in self.image_folder.iterdir()
            if filename.suffix.lower() == ".jpg"
            and self.annotations.__contains__(filename.name)
        ]
        # we use tensor instead of PIL.Image for training
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        annotation = self.annotations[self.images[index].name]
        image, annotation = resize(image, self.image_shape, annotation)
        return self.transform(image), annotation

    def _read_annotation(self, annotation_folder):
        """Read Annotation XMLs"""
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
            self.annotations[image_name] = torch.Tensor(annotation)


##########################################################
# Transforms
##########################################################


def resize(image, target_size, boxes, dtype=torch.half):
    """resize image and annotation box"""
    # Size of image
    image_height, image_width = image.size
    width, height = target_size

    # resize
    scale = min(width / image_width, height / image_height)
    new_width = int(image_width * scale)
    new_height = int(image_height * scale)
    delta_x = (width - new_width) // 2
    delta_y = (height - new_height) // 2

    # Resize the image to fit our "perception"
    image = image.resize((new_width, new_height))
    new_image = Image.new("RGB", target_size, (0, 0, 0))
    new_image.paste(image, (delta_x, delta_y))
    image_tensor = torch.Tensor(new_image, dtype=dtype)

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
        # drop the boxes if any dimension collapse to zero, should not happen
        box_w = boxes[:, 2] - boxes[:, 0]
        box_h = boxes[:, 3] - boxes[:, 1]
        boxes = boxes[torch.logical_and(box_w > 1, box_h > 1)]

    return image_tensor, boxes


if __name__ == "__main__":
    data = IsicDataSet("./dataset/input", "./dataset/annotation", ["lesion"])

    for i in range(1, 2):
        print(f"image: {data[i][0]} annotation: {data[i][1]}")
