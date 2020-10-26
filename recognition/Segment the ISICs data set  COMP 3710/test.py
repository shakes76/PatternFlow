import os
import tensorflow as tf
from model import Unet
import cv2
import argparse
import numpy as np
from tensorflow.keras import backend as K

tf.config.experimental.list_physical_devices('GPU')

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)


def preprocess(x, y):
    """图片预处理"""
    x = tf.cast(x, dtype=tf.float32) / 255
    y = tf.cast(y, dtype=tf.int16) / 255
    return x, y

def get_image(img_dir, size=(256, 256), mask=False):
    """获得图片"""
    imgs = []
    for i in os.listdir(img_dir):
        if mask:
            img = cv2.imread(os.path.join(img_dir, i), cv2.IMREAD_GRAYSCALE)
        else :
            img = cv2.imread(os.path.join(img_dir, i))
        img = cv2.resize(img, size)
        imgs.append(img)
    return np.array(imgs)

def get_db(data_dir, batch_size=32):
    x_test = get_image(os.path.join(data_dir, "Test_Data"))
    y_test = get_image(os.path.join(data_dir, "Test_GroundTruth"), mask=True)
    y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], y_test.shape[2], 1)
    db_t = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    db_t = db_t.map(preprocess).batch(batch_size=batch_size)
    return db_t

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        type=str,
                        default = "datsets",
                        help = "数据集地址",
                        )
    parser.add_argument("--workers",
                        type = int,
                        default = 8,
                        help = "number of workers",
                        )
    parser.add_argument("--batch_size",
                        type = int,
                        default = 32,
                        help = "batch size",
                        )
    parser.add_argument('--model',
                        default='weight/ep010-val_loss0.456-val_acc0.807',
                        type=str,
                        help='model path')
    args = parser.parse_args()

    model = Unet()
    model.build(input_shape=(None, 256, 256, 3))
    model.load_weights(args.model)
    db_test = get_db(args.data_dir, args.batch_size)

    sum_dice = 0
    i = 0
    for x, y in db_test:
        i += 1
        out = model.predict(x)
        sum_dice += dice_coef(y, out)
    print(sum_dice / i)

