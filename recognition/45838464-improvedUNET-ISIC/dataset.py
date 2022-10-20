import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.io import read_image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import cv2 

class ISICDataSet(Dataset):




    def __init__(self, ISIC_labels, ISIC_img_dir, transform=None, target_transform=None):
        self.ISIC_labels = ISIC_labels
        self.ISIC_img_dir = ISIC_img_dir
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.ISIC_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.ISIC_img_dir, self.ISIC_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.ISIC_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            image = self.target_transform(image)
        return image, label

class DataLoader(DataLoader):
    pass

def png2bw(png):
    grayImage = cv2.cvtColor(png, cv2.COLOR_BGR2GRAY)
    (thresh, bwImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
    
    return bwImage

def bBoxCoords(mask):

    # image size
    height, width = mask.shape
    # segmentation 
    countours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # bounding box in cv2 coordinates
    x,y,w,h = cv2.boundingRect(countours[0])
    # yolo coordinates
    x_rel =  (x + round(w/2)) / width
    y_rel = (y + round(h/2)) / height
    w_rel = w/width
    h_rel = h/height

    return (x_rel, y_rel, w_rel, h_rel)

def generate_labels(load_path, save_path):
    for mask in sorted(os.listdir(load_path)):
        
        # convert to binary matrix
        bw = png2bw(mask)
        # find bounding box
        bbox = bBoxCoords(bw)
        # generate
        new_row = {'object-class':0, 'X': x, 'Y': y, 'width': w, 'height': h}
        df = pd.DataFrame(data=new_row, index=[0])
        df.to_csv(output+'/'+filename[0:-3]+ 'txt',sep=' ',header=False, index=False)
        cv2.imwrite(output+'/'+filename,draw_image)
    



    # with_box = cv2.rectangle(mask, (int(x_rel*width), int(y_rel*height)), (int(x_rel*width + 20), int(y_rel*height + 20)), (100,0,0), 4)
    # with_box = cv2.rectangle(mask, (x_center,y_center), (x_center + 20, y_center + 20), (100,0,0), 4)
    # return with_box





# Load and save data 
# Data 


# clean data, augment, preprocessing
if __name__ == "__main__":
    train_images_dir = "./data/ISIC-2017_Training_Part1_GroundTruth"
    train_ground_truth = "./data/ISIC-2017_Training_Data"
     = cv2.imread("./data/ISIC-2017_Training_Part1_GroundTruth/ISIC_0000000_segmentation.png")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()