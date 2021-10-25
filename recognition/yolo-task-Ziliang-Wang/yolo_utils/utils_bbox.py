import torch
from torchvision.ops import nms
import numpy as np

# code is referenced from https://github.com/bubbliiiing/yolo3-pytorch/blob/master/utils/utils_bbox.py
# some adjusted small parts
class DecodeBox():
    """
    Adjust the anchor box
    """

    def __init__(self, anchors, num_classes, input_shape):
        super().__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_num = 5 + num_classes
        self.input_shape = input_shape

        self.anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    def decode_box(self, inputs):
        outputs = []
        # there are three (three upsampling)different size of feature layer
        for index, input in enumerate(inputs):
            bs, height, width = input.size(0), input.size(2), input.size(3)
            
            # Get the stride, for the 13*13 feature layer, every feature point correspond 32 pixes
            # in the original picture
            # eg. input = 416*416, stride height = 416/13 = 32
            scaled_pixel_height = self.input_shape[0] / height
            scaled_pixel_width = self.input_shape[1] / width

            # scale the anchor box to correspond size
            # the original anchor size is for 416*416
            # mask [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
            # anchors = [[10., 13.],
            #            [16., 30.],
            #            [33., 23.],
            #            [30., 61.],
            #            [62., 45.],
            #            [59., 119.],
            #            [116., 90.],
            #            [156., 198.],
            #            [373., 326.]]

            scaled_anchor = []

            for w, h in self.anchors[self.anchor_mask[index]]:
                scaled_anchor.append((w / scaled_pixel_width, h / scaled_pixel_height)
                                     )
            # adjust the dim
            prediction = input.view(bs, 3,
                                    self.bbox_num, height, width).permute(0, 1, 3, 4, 2).contiguous()

            # adjust the center anchor box with in 0-1
            x = torch.sigmoid(prediction[..., 0])
            y = torch.sigmoid(prediction[..., 1])
            w = prediction[..., 2]
            h = prediction[..., 3]
            conf = torch.sigmoid(prediction[..., 4])
            pred_cls = torch.sigmoid(prediction[..., 5:])

            FloatTensor = torch.cuda.FloatTensor
            LongTensor = torch.cuda.LongTensor

            # Generate grid, center of anchor box, in the upper left corner of the grid
            grid_x = torch.linspace(0, width - 1, width).repeat(height, 1).repeat(
                bs * len(self.anchor_mask[index]), 1, 1).view(x.shape).type(FloatTensor)
            grid_y = torch.linspace(0, height - 1, height).repeat(width, 1).t().repeat(
                bs * len(self.anchor_mask[index]), 1, 1).view(y.shape).type(FloatTensor)

            # Generate width  and height of anchor box
            width_anchor = FloatTensor(scaled_anchor).index_select(1, LongTensor([0])).repeat(bs, 1).repeat(1, 1,
                                                                                                            height * width).view(
                w.shape)
            height_anchor = FloatTensor(scaled_anchor).index_select(1, LongTensor([1])).repeat(bs, 1).repeat(1, 1,
                                                                                                             height * width).view(
                h.shape)

            # Adjust the anchor box according to the regression results
            # offset from the upper left corner of the grid
            pred_boxes = FloatTensor(prediction[..., :4].shape)
            pred_boxes[..., 0] = x.data + grid_x
            pred_boxes[..., 1] = y.data + grid_y
            pred_boxes[..., 2] = torch.exp(w.data) * width_anchor
            pred_boxes[..., 3] = torch.exp(h.data) * height_anchor

            # Normalize the output
            _scale = torch.Tensor([width, height, width, height]).type(FloatTensor)
            output = torch.cat((pred_boxes.view(bs, -1, 4) / _scale,
                                conf.view(bs, -1, 1), pred_cls.view(bs, -1, self.num_classes)), -1)
            outputs.append(output.data)
        return outputs

    def yolo_correct_boxes(self, box_xy, box_wh, input_shape, image_shape, letterbox_image):
        """
        return the correct bbox information
        """

        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        if letterbox_image:
            new_shape = np.round(image_shape * np.min(input_shape / image_shape))
            offset = (input_shape - new_shape) / 2. / input_shape
            scale = input_shape / new_shape

            box_yx = (box_yx - offset) * scale
            box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]],
                               axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes

    def non_max_suppression(self, prediction, num_classes, input_shape, image_shape, letterbox_image, conf_thres,
                            nms_thres):
        """
        Starting from the maximum probability rectangle F,
        judge whether the overlap IOU between A~E and F is greater than a specified threshold;
        """
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):

            class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)

            conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()

            image_pred = image_pred[conf_mask]
            class_conf = class_conf[conf_mask]
            class_pred = class_pred[conf_mask]
            if not image_pred.size(0):
                continue

            detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)

            unique_labels = detections[:, -1].cpu().unique()

            if prediction.is_cuda:
                unique_labels = unique_labels.cuda()
                detections = detections.cuda()

            for c in unique_labels:
                detections_class = detections[detections[:, -1] == c]

                keep = nms(
                    detections_class[:, :4],
                    detections_class[:, 4] * detections_class[:, 5],
                    nms_thres
                )
                max_detections = detections_class[keep]

                output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))

            if output[i] is not None:
                output[i] = output[i].cpu().numpy()
                box_xy, box_wh = (output[i][:, 0:2] + output[i][:, 2:4]) / 2, output[i][:, 2:4] - output[i][:, 0:2]
                output[i][:, :4] = self.yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
        return output
