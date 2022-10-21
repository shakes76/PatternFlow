import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LeakyReLU
from keras.models import Model
import keras.backend as K
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
import os


class Yolo_Reshape(tf.keras.layers.Layer):
    def __init__(self, target_shape):
        super(Yolo_Reshape, self).__init__()
        self.target_shape = tuple(target_shape)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'target_shape': self.target_shape
        })
        return config

    def call(self, input):
        # grids 7x7
        S = [self.target_shape[0], self.target_shape[1]]
        # classes
        C = 2
        # no of bounding boxes per grid
        B = 2

        idx1 = S[0] * S[1] * C
        idx2 = idx1 + S[0] * S[1] * B

        # class probabilities
        class_probs = K.reshape(input[:, :idx1], (K.shape(input)[0],) + tuple([S[0], S[1], C]))
        class_probs = K.softmax(class_probs)

        # confidence
        confs = K.reshape(input[:, idx1:idx2], (K.shape(input)[0],) + tuple([S[0], S[1], B]))
        confs = K.sigmoid(confs)

        # boxes
        boxes = K.reshape(input[:, idx2:], (K.shape(input)[0],) + tuple([S[0], S[1], B * 4]))
        boxes = K.sigmoid(boxes)

        outputs = K.concatenate([class_probs, confs, boxes])
        return outputs


lrelu = LeakyReLU(alpha=0.1)

def block_1(inputs):
  conv = Conv2D(64, (7, 7), strides=(1,1), activation=lrelu, padding='same')(inputs)
  conv = MaxPooling2D(pool_size = (2,2), strides=(2,2), padding='same')(conv)
  print(conv.shape)
  return conv

def block_2(conv):
  conv = Conv2D(192, (3, 3), activation=lrelu, padding='same')(conv)
  conv = MaxPooling2D(pool_size = (2,2), strides=(2,2), padding='same')(conv)
  print(conv.shape)
  return conv

def block_3(conv):
  conv = Conv2D(128, (1, 1), activation=lrelu, padding='same')(conv)
  conv = Conv2D(256, (3, 3), activation=lrelu, padding='same')(conv)
  conv = Conv2D(256, (1, 1), activation=lrelu, padding='same')(conv)
  conv = Conv2D(512, (3, 3), activation=lrelu, padding='same')(conv)
  conv = MaxPooling2D(pool_size = (2,2), strides=(2,2), padding='same')(conv)
  print(conv.shape)
  return conv

def block_4(conv):
  for i in range(4):
    conv = Conv2D(256, (1, 1), activation=lrelu, padding='same')(conv)
    conv = Conv2D(512, (3, 3), activation=lrelu, padding='same')(conv)
  conv = Conv2D(512, (1, 1), activation=lrelu, padding='same')(conv)
  conv = Conv2D(1024, (3, 3), activation=lrelu, padding='same')(conv)
  conv = MaxPooling2D(pool_size = (2,2), strides=(2,2), padding='same')(conv)
  print(conv.shape)
  return conv

def block_5(conv):
  for i in range(2):
    conv = Conv2D(512, (1, 1), activation=lrelu, padding='same')(conv)
    conv = Conv2D(1024, (3, 3), activation=lrelu, padding='same')(conv)
  conv = Conv2D(1024, (3, 3), activation=lrelu, padding='same')(conv)
  conv = Conv2D(1024, (3, 3), activation=lrelu, padding='same')(conv)
  conv = Conv2D(1024, (3, 3), strides=(2,2), padding='same')(conv)
  print(conv.shape)
  return conv

def block_6(conv):
  conv = Conv2D(1024, (3, 3), activation=lrelu, padding='same')(conv)
  conv = Conv2D(1024, (3, 3), activation=lrelu, padding='same')(conv)
  print(conv.shape)
  return conv

def block_7(conv):
  conv = Flatten()(conv)
  conv = Dense(512)(conv)
  conv = Dense(1024)(conv)
  conv = Dropout(0.5)(conv)
  conv = Dense(588, activation='sigmoid')(conv)
  print(conv.shape)
  output = Yolo_Reshape(target_shape=(7,7,12))(conv)
  print(output.shape)
  return output


# mcp_save = ModelCheckpoint('weight.hdf5', save_best_only=True, monitor='val_loss', mode='min')
checkpoit_path = ("SavedModel")
checkpoint_dir = os.path.dirname(checkpoit_path)

cp_callback = ModelCheckpoint(checkpoit_path, save_weights_only=True, verbose=1)



class CustomLearningRateScheduler(keras.callbacks.Callback):

    def __init__(self, schedule):
        super(CustomLearningRateScheduler, self).__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate from model's optimizer.
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.schedule(epoch, lr)
        # Set the value back to the optimizer before this epoch starts
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        print("\nEpoch %05d: Learning rate is %6.4f." % (epoch, scheduled_lr))


LR_SCHEDULE = [
    # (epoch to start, learning rate) tuples
    (0, 0.01),
    (20, 0.001),
    (40, 0.0001),
]


def lr_schedule(epoch, lr):
    """Helper function to retrieve the scheduled learning rate based on epoch."""
    if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
        return lr
    for i in range(len(LR_SCHEDULE)):
        if epoch == LR_SCHEDULE[i][0]:
            return LR_SCHEDULE[i][1]
    return lr


def xywh2minmax(xy, wh):
    xy_min = xy - wh / 2
    xy_max = xy + wh / 2

    return xy_min, xy_max


def iou(pred_mins, pred_maxes, true_mins, true_maxes):
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
    # Dynamic implementation of conv dims for fully convolutional model.
    conv_dims = K.shape(feats)[1:3]  # assuming channels last
    # In YOLO the height index is the inner most iteration.
    conv_height_index = K.arange(0, stop=conv_dims[0])
    conv_width_index = K.arange(0, stop=conv_dims[1])
    conv_height_index = K.tile(conv_height_index, [conv_dims[1]])

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



def yolo_head(feats):
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

def yolo_loss(y_true, y_pred):
    label_class = y_true[..., :2]  # ? * 7 * 7 * 2 (c1,c2)
    label_box = y_true[..., 2:6]  # ? * 7 * 7 * 4 (x,y,w,h)
    response_mask = y_true[..., 6]  # ? * 7 * 7 (r)
    response_mask = K.expand_dims(response_mask)  # ? * 7 * 7 * 1

    predict_class = y_pred[..., :2]  # ? * 7 * 7 * 2 (c1,c2)
    predict_trust = y_pred[..., 2:4]  # ? * 7 * 7 * 2 (x,y)
    predict_box = y_pred[..., 4:]  # ? * 7 * 7 * 3 (w,h,r)

    _label_box = K.reshape(label_box, [-1, 7, 7, 1, 4]) # (has to be right)
    _predict_box = K.reshape(predict_box, [-1, 7, 7, 2, 4]) # (has to be right)

    label_xy, label_wh = yolo_head(_label_box)  # ? * 7 * 7 * 1 * 2, ? * 7 * 7 * 1 * 2 (has to be right)
    label_xy = K.expand_dims(label_xy, 3)  # ? * 7 * 7 * 1 * 1 * 2 (has to be right)
    label_wh = K.expand_dims(label_wh, 3)  # ? * 7 * 7 * 1 * 1 * 2 (has to be right)
    label_xy_min, label_xy_max = xywh2minmax(label_xy, label_wh)  # ? * 7 * 7 * 1 * 1 * 2, ? * 7 * 7 * 1 * 1 * 2 (has to be right)

    predict_xy, predict_wh = yolo_head(_predict_box)  # ? * 7 * 7 * 2 * 2, ? * 7 * 7 * 2 * 2 (has to be right)
    predict_xy = K.expand_dims(predict_xy, 4)  # ? * 7 * 7 * 2 * 1 * 2 (has to be right)
    predict_wh = K.expand_dims(predict_wh, 4)  # ? * 7 * 7 * 2 * 1 * 2 (has to be right)
    predict_xy_min, predict_xy_max = xywh2minmax(predict_xy, predict_wh)  # ? * 7 * 7 * 2 * 1 * 2, ? * 7 * 7 * 2 * 1 * 2 (has to be right)

    iou_scores = iou(predict_xy_min, predict_xy_max, label_xy_min, label_xy_max)  # ? * 7 * 7 * 2 * 1 (has to be right)
    best_ious = K.max(iou_scores, axis=4)  # ? * 7 * 7 * 2 (has to be right)
    best_box = K.max(best_ious, axis=3, keepdims=True)  # ? * 7 * 7 * 1 (has to be right)

    box_mask = K.cast(best_ious >= best_box, K.dtype(best_ious))  # ? * 7 * 7 * 2 (has to be right)

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