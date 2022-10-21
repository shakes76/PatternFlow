import numpy as np
import cv2 as cv
import pandas as pd
from glob import glob
from tqdm import tqdm
from skimage.measure import label, regionprops, find_contours
import os
import re

""" Creating a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

""" Convert a mask to border image """
def mask_to_border(mask):
    h, w = mask.shape
    border = np.zeros((h, w))

    contours = find_contours(mask, 128)
    for contour in contours:
        for c in contour:
            x = int(c[0])
            y = int(c[1])
            border[x][y] = 255

    return border

def mask_to_bbox(mask):
    bboxes = []

    mask = mask_to_border(mask)
    lbl = label(mask)
    props = regionprops(lbl)
    for prop in props:
        x1 = prop.bbox[1]
        y1 = prop.bbox[0]

        x2 = prop.bbox[3]
        y2 = prop.bbox[2]

        bboxes.append([x1, y1, x2, y2])

    return bboxes

def parse_mask(mask):
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)
    return mask



def bbox_to_list(im_path, mask_path, cls_path):
  images = sorted(glob(os.path.join(im_path, "*")))
  masks = sorted(glob(os.path.join(mask_path, "*")))
  bboxes_list = []
  for x, y in tqdm(zip(images, masks), total=len(images)):
    """ Extract the name """
    name = x.split("/")[-1].split(".")[0]

    """ Read image and mask """
    x = cv.imread(x, cv.IMREAD_COLOR)
    y = cv.imread(y, cv.IMREAD_GRAYSCALE)

    """ Detecting bounding boxes """
    bboxes = mask_to_bbox(y)
    bboxes_list.append(bboxes)

  """ Taking only one biggest bounding box """
  for i in range(len(bboxes_list)):
    index = 0
    max_el = 0
    for j in range(len(bboxes_list[i])):
      diff = (bboxes_list[i][j][2] - bboxes_list[i][j][0])+(bboxes_list[i][j][3] - bboxes_list[i][j][1])
      if(diff > max_el):
        max_el = diff
        index = j
    bboxes_list[i] = bboxes_list[i][index]

  """ Adding class label to bbox list"""
  classes = pd.read_csv(cls_path)
  classes = np.asarray(classes['melanoma'])
  for i in range(2000):
    bboxes_list[i].append(int(classes[i]))

  return bboxes_list



def savetxt_compact(fname, x, fmt="%.0g", delimiter=','):
    with open(fname, 'w') as fh:
        for row in x:
            line = delimiter.join("0" if value == 0 else fmt % value for value in row)
            fh.write(line + '\n')




def img_path_list (img_path):
  path_list = []
  for root, dirs, files in os.walk(os.path.abspath(img_path+"/")):
      for file in files:
          path_list.append(os.path.join(root, file))
  def atoi(text):
    return int(text) if text.isdigit() else text
  def natural_keys(text):
      return [ atoi(c) for c in re.split('(\d+)',text) ]
  path_list.sort(key=natural_keys)

  return path_list



def read(image_path, label):
    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = cv.resize(image, (448, 448))
    image = image / 255.

    label_matrix = np.zeros([7, 7, 12])
    for l in label:
        l = l.split(',')
        l = np.array(l, dtype=np.int) #converts string to int array [x1,y1,x2,y2]
        xmin = l[0]
        ymin = l[1]
        xmax = l[2]
        ymax = l[3]
        cls = l[4]
        x = ((xmin + xmax) / 2 / 448)
        y = ((ymin + ymax) / 2 / 448)
        w = ((xmax - xmin) / 448)
        h = ((ymax - ymin) / 448)
        # loc = [7 * x, 7 * y]
        # loc_i = int(loc[1])
        # loc_j = int(loc[0])
        # y = loc[1] - loc_i
        # x = loc[0] - loc_j

        if label_matrix[0, 0, 6] == 0:
            label_matrix[0, 0, cls] = 1
            label_matrix[0, 0, 2:6] = [x, y, w, h]
            label_matrix[0, 0, 6] = 1  # response

    return image, label_matrix


train_image = []
train_label = []
def load_data (X, Y):
    for i in range(0, len(X)):
      img_path = X[i]
      label = Y[i]
      image, label_matrix = read(img_path, label)
      train_image.append(image)
      train_label.append(label_matrix)
      x = np.array(train_image)
      y = np.array(train_label)
    return x, y

"""Split the data into train, validation and test sets"""

def tr_val_ts_split (x, y):
    train_threshold = int(x.shape[0]/100*90)
    test_threshold = int(x.shape[0]/100*95)

    train_ind = []
    for i in range(train_threshold):
      train_ind.append(i)

    test_ind = []
    for i in range(train_threshold, test_threshold):
      test_ind.append(i)

    val_ind = []
    for i in range(test_threshold, x.shape[0]):
      val_ind.append(i)

    x_train, x_test, x_val = x[train_ind], x[test_ind], x[val_ind]
    y_train, y_test, y_val = y[train_ind], y[test_ind], y[val_ind]

    return x_train, x_test, x_val, y_train, y_test, y_val
