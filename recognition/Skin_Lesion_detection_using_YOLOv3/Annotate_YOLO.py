"""
Driver script to import and annotate the ISICs dataset.

@author Lalith Veerabhadrappa Badiger
@email l.badiger@uqconnect.edu.au
"""

# Import required libraries
import cv2
from glob import glob

# Provide the filepath for ISICs segmentation and train dataset
segmentation = 'ISIC2018_Task1-2_Training_Data/ISIC2018_Task1_Training_GroundTruth_x2'
train = 'ISIC2018_Task1-2_Training_Data/ISIC2018_Task1-2_Training_Input_x2'


def get_bbox(img):
    """
    Returns label (i.e. index of the class name) and (x,y,width,height) coordinates. 
    These are created from the given segmentation images.
    """
    label = 0 # Only one class which has index = 0
    
    # convert to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # threshold
    thresh = cv2.threshold(gray,128,255,cv2.THRESH_BINARY)[1]
    
    # get contours
    result = img.copy()
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    
    for cntr in contours:
        x,y,w,h = cv2.boundingRect(cntr)
        
    return label,x,y,w,h


def normalize(x, y, num_rows, num_cols):
    """
    Returns the normalized coordinates in the range 0 to 1.
    This is done to make the coordinates invariant to the dimensions of the image.
    """
    x = x/(num_cols)
    y = y/(num_rows)
    return x, y


def normalized_coordinates(x, y, w, h, img):
    """
    Returns the true annotations of bounding boxes in YOLO format.
    """
    rows, cols = img.shape[:2] # Get the dimensions of the image
    x_centre, y_centre = (w/2)+x, (h/2)+y # Returns centre of the bounding box
    x_nrm, y_nrm = normalize(x_centre, y_centre, rows, cols)
    w_nrm, h_nrm = normalize(w, h, rows, cols)
    return x_nrm, y_nrm, w_nrm, h_nrm


def write_txt():
    """
    Writes to text files, the true annotations of bounding boxes in YOLO format. 
    There will be one text file for every image which acts as a true label.
    """
    for i in glob(segmentation + '/*.png'):
        img = cv2.imread(i)
        label,x,y,w,h = get_bbox(img)
        x,y,w,h = normalized_coordinates(x,y,w,h,img)
        text = f'{label} {x} {y} {w} {h}'
        with open(train + '/' + i[70:82] + '.txt', 'w') as f:
            f.write(text)


write_txt()
