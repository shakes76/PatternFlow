import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D, Flatten, Dense, Reshape
from tensorflow.keras.models import Model

import keras.backend as K

class YoloModel():
    
    def __init__(self, n_classes, n_boxes, n_cells, img_width, img_height):
        """ Create a new Yolo Model instance.
        
        Parameters:
            n_classes (int): 
            n_boxes (int): 
            n_cells (int): 
            img_wdith (int): 
            img_height (int): 
        """
        self.n_classes = n_classes
        self.n_boxes = n_boxes
        self.n_cells = n_cells
        self.img_width = img_width
        self.img_height = img_height
        
        self.model = self.modelArchitecture()
        
    def xywh2minmax(xy, wh):
        """
        
        
        """
        xy_min = xy - wh / 2
        xy_max = xy + wh / 2

        return xy_min, xy_max


    def iou(pred_mins, pred_maxes, true_mins, true_maxes):
        """
        
        
        """
        intersect_mins = K.maximum(pred_mins, true_mins)
        intersect_maxes = K.minimum(pred_maxes, true_maxes)
        intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        pred_wh = pred_maxes - pred_mins
        true_wh = true_maxes - true_mins
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
        true_areas = true_wh[..., 0] * true_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = intersect_areas / union_areas

        return iou_scores


    def yolo_head(feats):
        """
        
        
        """
        # Dynamic implementation of conv dims for fully convolutional model.
        conv_dims = K.shape(feats)[1:3]  # assuming channels last
        # In YOLO the height index is the inner most iteration.
        conv_height_index = K.arange(0, stop=conv_dims[0])
        conv_width_index = K.arange(0, stop=conv_dims[1])
        conv_height_index = K.tile(conv_height_index, [conv_dims[1]])

        # TODO: Repeat_elements and tf.split doesn't support dynamic splits.
        # conv_width_index = K.repeat_elements(conv_width_index, conv_dims[1], axis=0)
        conv_width_index = K.tile(
            K.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
        conv_width_index = K.flatten(K.transpose(conv_width_index))
        conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))
        conv_index = K.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
        conv_index = K.cast(conv_index, K.dtype(feats))

        conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 2]), K.dtype(feats))

        box_xy = (feats[..., :2] + conv_index) / conv_dims * 448
        box_wh = feats[..., 2:4] * 448

        return box_xy, box_wh
        
    def yoloLoss(self, y_true, y_pred):
        """ Yolo Model Loss function specified by You Only Look Once paper
        
        Parameters:
            y_true (tf.Dataset): A (batchSize, n_cells, n_cells, ) tensor containing the true bounding boxes.
            y_pred (tf.Dataset): A () tensor containing the predicted bounding boxes.
            
        Return:
            tf.Dataset: A () tensor containing the loss for each cell
        
        """
        
        label_class = y_true[..., :20]  # ? * 7 * 7 * 20
        label_box = y_true[..., 20:24]  # ? * 7 * 7 * 4
        response_mask = y_true[..., 24]  # ? * 7 * 7
        response_mask = K.expand_dims(response_mask)  # ? * 7 * 7 * 1

        predict_class = y_pred[..., :20]  # ? * 7 * 7 * 20
        predict_trust = y_pred[..., 20:22]  # ? * 7 * 7 * 2
        predict_box = y_pred[..., 22:]  # ? * 7 * 7 * 8

        _label_box = K.reshape(label_box, [-1, 7, 7, 1, 4])
        _predict_box = K.reshape(predict_box, [-1, 7, 7, 2, 4])

        label_xy, label_wh = yolo_head(_label_box)  # ? * 7 * 7 * 1 * 2, ? * 7 * 7 * 1 * 2
        label_xy = K.expand_dims(label_xy, 3)  # ? * 7 * 7 * 1 * 1 * 2
        label_wh = K.expand_dims(label_wh, 3)  # ? * 7 * 7 * 1 * 1 * 2
        label_xy_min, label_xy_max = xywh2minmax(label_xy, label_wh)  # ? * 7 * 7 * 1 * 1 * 2, ? * 7 * 7 * 1 * 1 * 2

        predict_xy, predict_wh = yolo_head(_predict_box)  # ? * 7 * 7 * 2 * 2, ? * 7 * 7 * 2 * 2
        predict_xy = K.expand_dims(predict_xy, 4)  # ? * 7 * 7 * 2 * 1 * 2
        predict_wh = K.expand_dims(predict_wh, 4)  # ? * 7 * 7 * 2 * 1 * 2
        predict_xy_min, predict_xy_max = xywh2minmax(predict_xy, predict_wh)  # ? * 7 * 7 * 2 * 1 * 2, ? * 7 * 7 * 2 * 1 * 2

        iou_scores = iou(predict_xy_min, predict_xy_max, label_xy_min, label_xy_max)  # ? * 7 * 7 * 2 * 1
        best_ious = K.max(iou_scores, axis=4)  # ? * 7 * 7 * 2
        best_box = K.max(best_ious, axis=3, keepdims=True)  # ? * 7 * 7 * 1

        box_mask = K.cast(best_ious >= best_box, K.dtype(best_ious))  # ? * 7 * 7 * 2

        no_object_loss = 0.5 * (1 - box_mask * response_mask) * K.square(0 - predict_trust)
        object_loss = box_mask * response_mask * K.square(1 - predict_trust)
        confidence_loss = no_object_loss + object_loss
        confidence_loss = K.sum(confidence_loss)

        class_loss = response_mask * K.square(label_class - predict_class)
        class_loss = K.sum(class_loss)

        _label_box = K.reshape(label_box, [-1, 7, 7, 1, 4])
        _predict_box = K.reshape(predict_box, [-1, 7, 7, 2, 4])

        label_xy, label_wh = yolo_head(_label_box)  # ? * 7 * 7 * 1 * 2, ? * 7 * 7 * 1 * 2
        predict_xy, predict_wh = yolo_head(_predict_box)  # ? * 7 * 7 * 2 * 2, ? * 7 * 7 * 2 * 2

        box_mask = K.expand_dims(box_mask)
        response_mask = K.expand_dims(response_mask)

        box_loss = 5 * box_mask * response_mask * K.square((label_xy - predict_xy) / 448)
        box_loss += 5 * box_mask * response_mask * K.square((K.sqrt(label_wh) - K.sqrt(predict_wh)) / 448)
        box_loss = K.sum(box_loss)

        loss = confidence_loss + class_loss + box_loss
        
        return loss
        
    def modelArchitecture(self):
        """ Defines Yolo model network
        Described in the paper You Only Look Once
        

        Return:
            tf.keras.models.Model: CNN defined by Yolo architecture

        """
        
        inputs = Input(shape=(self.img_width, self.img_height, 3))
        
        # First Layer
        x = Conv2D(filters=64, kernel_size=(7,7), strides=(1,1), padding='same')(inputs)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D((2,2), strides=(2,2))(x)
        
        # Second Layer
        x = Conv2D(192, (3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D((2,2), strides=(2,2))(x)
        
        # Third Layer
        x = Conv2D(128, (1,1), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        x = Conv2D(256, (3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        x = Conv2D(256, (1,1), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        x = Conv2D(512, (3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        x = MaxPooling2D((2,2), strides=(2,2))(x)
        
        # Fourth Layer
        x = Conv2D(256, (1,1), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        x = Conv2D(512, (3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        x = Conv2D(256, (1,1), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        x = Conv2D(512, (3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        x = Conv2D(256, (1,1), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        x = Conv2D(512, (3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        x = Conv2D(256, (1,1), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        x = Conv2D(512, (3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        x = Conv2D(512, (1,1), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        x = Conv2D(1024, (3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        x = MaxPooling2D((2,2), strides=(2,2))(x)
        
        # Fifth Layer
        x = Conv2D(512, (1,1), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        x = Conv2D(1024, (3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        x = Conv2D(512, (1,1), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        x = Conv2D(1024, (3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        x = Conv2D(1024, (3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        x = Conv2D(1024, (3,3), strides=(2,2), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        # Sixth Layer
        x = Conv2D(1024, (3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        x = Conv2D(1024, (3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        # Final Layer
        x = Flatten()(x)
        x = Dense(4096)(x)
        x = Dense(self.n_cells * self.n_cells * (self.n_boxes * 5 + self.n_classes))(x)
        outputs = Reshape((self.n_cells, self.n_cells, (self.n_boxes * 5 + self.n_classes)))(x)
        
        yolo_model = Model(inputs, outputs)
        
        return yolo_model
    
    def compileModel(self): 
        self.model.compile(optimizer='adam',
                          loss=self.yoloLoss,
                          metrics=[tf.keras.metrics.IoU(num_classes=self.n_classes, target_class_ids=[0])])