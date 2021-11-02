import os
from typing import List
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset


class IsicDataSet(Dataset):
    def __init__(
        self, image_folder: str, annotation_folder: str, classes: List[str]
    ) -> None:
        super().__init__()
        self.classes = classes
        self._read_annotation(annotation_folder)
        self.images = [
            os.path.join(image_folder, filename)
            for filename in os.listdir(image_folder)
            if filename.endswith(".jpg") and self.annotations.__contains__(filename)
        ]
        self._resize()

    def __len__(self):
        len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        annotation = self.annotations[self.images[index]]
        return image, annotation

    def _read_annotation(self, annotation_folder):
        self.annotations = {}
        for filename in os.listdir(annotation_folder):
            xml = ET.parse(os.path.join(annotation_folder, filename)).getroot()
            image_name = xml.subelement("filename").text
            annotation = []
            for obj in xml.iter("object"):
                cls = obj.find("name").text
                if cls not in self.classes:
                    continue
                cls_id = self.classes.index(cls)
                xmlbox = obj.find("bndbox")
                bbox = [
                    int(float(xmlbox.find("xmin").text)),
                    int(float(xmlbox.find("ymin").text)),
                    int(float(xmlbox.find("xmax").text)),
                    int(float(xmlbox.find("ymax").text)),
                    cls_id,
                ]
                annotation.append(bbox)
            self.annotations[image_name] = annotation
