from pathlib import Path
from typing import List
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


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
            if filename.suffix.lower()==".jpg"
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


if __name__ == "__main__":
    data = IsicDataSet("./dataset/input", "./dataset/annotation", ["lesion"])

    for i in range(1, 2):
        print(f"image: {data[i][0]} annotation: {data[i][1]}")
