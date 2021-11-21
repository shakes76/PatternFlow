import numpy as np
import torch

import train
from yolo_utils.utils import (cvtColor, preprocess_input,
                              resize_image)
from yolo_utils.utils_bbox import DecodeBox
from model.model import YoloBody
import driver
import cv2
import xml.dom.minidom


# Referenced from https://github.com/bubbliiiing/yolo3-pytorch
class YoloDetect(object):

    def __init__(self, weight_path):

        self.weights_path, self.anchors_mask, self.input_shape, self.class_names, self.num_classes, self.anchors, self.num_anchors, self.confidence, self.nms_iou, self.letterbox_image = driver.get_variable()
        self.weights_path = weight_path
        self.bbox_util = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]),
                                   )

        self.net = YoloBody(self.anchors_mask, self.num_classes)
        self.net.load_state_dict(torch.load(self.weights_path, map_location=train.device))
        self.net = self.net.eval()
        print('{} model, anchors, and classes loaded.'.format(self.weights_path))
        self.net = self.net.cuda()

    def detect_image(self, image, img_line, img_name, detect_image=True):
        """
        Detect the image
        """
        image_shape = np.array(np.shape(image)[0:2])
        # The image is converted into RGB
        image = cvtColor(image)
        # Add gray bars to the image
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)

        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            images = images.cuda()

            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)

            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,
                                                         image_shape, self.letterbox_image, conf_thres=self.confidence,
                                                         nms_thres=self.nms_iou)

            if results[0] is None:
                if detect_image:
                    image = cv2.imread(img_line)
                    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                    cv2.imshow('Lesion detection', image)
                    cv2.waitKey(0)
                else:
                    return 0
                return 0

            top_label = np.array(results[0][:, 6], dtype='int32')
            top_conf = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]

        #   Image drawing
        for i, c in list(enumerate(top_label)):
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box

            ymin = max(0, np.floor(top).astype('int32'))
            xmin = max(0, np.floor(left).astype('int32'))
            ymax = min(image.size[1], np.floor(bottom).astype('int32'))
            xmax = min(image.size[0], np.floor(right).astype('int32'))
            root = xml.dom.minidom.parse(driver.xml_root + img_name[:-4] + ".xml").documentElement
            xmin_t = root.getElementsByTagName('xmin')[0].firstChild.data
            ymin_t = root.getElementsByTagName('ymin')[0].firstChild.data
            xmax_t = root.getElementsByTagName('xmax')[0].firstChild.data
            ymax_t = root.getElementsByTagName('ymax')[0].firstChild.data

            iou = self.compute_iou((xmin, ymin, xmax, ymax), (int(xmin_t), int(ymin_t), int(xmax_t), int(ymax_t)))
            print("iou: ", iou)
            if detect_image:
                image = cv2.imread(img_line)
                draw = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                cv2.putText(image, driver.class_names[0] + str(score), (xmin - 10, ymin - 10), cv2.FONT_HERSHEY_DUPLEX,
                            0.5,
                            (255, 0, 0), 1)
                resize = cv2.resize(draw, (1400, 900))
                cv2.imshow('Lesion detection', resize)
                cv2.waitKey(0)
            else:
                pass
            return iou

    def calculate_area(self, xmin, ymin, xmax, ymax):
        return (xmax - xmin) * (ymax - ymin)

    def compute_iou(self, box1, box2):
        # computing the two boxes area
        xmin, ymin, xmax, ymax = box1
        area1 = self.calculate_area(xmin, ymin, xmax, ymax)
        xmin, ymin, xmax, ymax = box2
        area2 = self.calculate_area(xmin, ymin, xmax, ymax)

        intersection = (min(box1[2], box2[2]) - max(box1[0], box2[0])) * (min(box1[3], box2[3]) - max(box1[1], box2[1]))
        union = area1 + area2 - intersection
        return (float(intersection) / float(union))
