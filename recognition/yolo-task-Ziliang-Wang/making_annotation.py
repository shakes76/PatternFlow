import os
import random
import xml.dom.minidom
from random import shuffle

xml_root = 'dataset/Annotations/'
JPEG_root = 'dataset/JPEGImages/'
random.seed(0)  # Ensure that the results of the division dataset can be reproduced
split_rate = 0.8


def make_annotation(xml_root, JPEG_root):
    """
    Make the train and test file txt file according the xml file.
    """
    files = os.listdir(xml_root)[1:]
    abs_path = os.path.dirname(os.path.abspath(__file__))

    coordinate = []
    for file in files:
        root = xml.dom.minidom.parse(xml_root + file).documentElement
        xmin = root.getElementsByTagName('xmin')[0].firstChild.data
        ymin = root.getElementsByTagName('ymin')[0].firstChild.data
        xmax = root.getElementsByTagName('xmax')[0].firstChild.data
        ymax = root.getElementsByTagName('ymax')[0].firstChild.data
        file_name = root.getElementsByTagName('filename')[0].firstChild.data
        truncated = root.getElementsByTagName('truncated')[0].firstChild.data
        coordinate.append((xmin, ymin, xmax, ymax, truncated, file_name))

    shuffle(coordinate)
    round(len(coordinate) * split_rate)
    train = coordinate[:round(len(coordinate) * split_rate)]
    test = coordinate[round(len(coordinate) * split_rate):]
    print(len(train))
    print(len(test))

    for i in range(len(train)):
        f = open(abs_path + "/train.txt", "a")
        f.write(
            abs_path + '/' + JPEG_root + train[i][-1] + ".jpg" + " " + train[i][0] + "," + train[i][1] + "," +
            train[i][2] + "," + train[i][3] + "," + train[i][4] + "\n")
        f.close()

    for i in range(len(test)):
        f = open(abs_path + "/test.txt", "a")
        f.write(
            abs_path + '/' + JPEG_root + test[i][-1] + ".jpg" + " " + test[i][0] + "," + test[i][1] + "," +
            test[i][2] + "," + test[i][3] + "," + test[i][4] + "\n")
        f.close()

    for i in range(len(train)):
        f = open(abs_path + "/train_image_name.txt", "a")
        f.write(
            train[i][-1] + ".jpg" + "\n")
        f.close()

    for i in range(len(test)):
        f = open(abs_path + "/test_image_name.txt", "a")
        f.write(
            test[i][-1] + ".jpg" + "\n")
        f.close()


make_annotation(xml_root, JPEG_root)
