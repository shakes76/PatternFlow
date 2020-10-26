import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers, callbacks
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

def dice_coef_loss(y_true, y_pred):
	return 1 - dice_coef(y_true, y_pred, smooth=1)

def preprocess(x, y):
    """图片预处理"""
    x = tf.cast(x, dtype=tf.float32) / 255
    y = tf.cast(y, dtype=tf.int16)
    return x, y

def get_image(img_dir, size=(256, 256), mask=False):
    """获得图片"""
    imgs = []
    for i in os.listdir(img_dir)[:100]:
        if mask:
            img = cv2.imread(os.path.join(img_dir, i), cv2.IMREAD_GRAYSCALE) / 255
            img[img > 0.5] = 1
            img[img <= 0.5] = 0
        else :
            img = cv2.imread(os.path.join(img_dir, i))
        img = cv2.resize(img, size)
        imgs.append(img)
    return np.array(imgs)

def get_db(data_dir, batch_size=32, shuffle=10000):
    """返回训练集和测试集"""
    x = get_image(os.path.join(data_dir, "Training_Data"))
    y = get_image(os.path.join(data_dir, "Training_GroundTruth"), mask=True)
    y = y.reshape(y.shape[0], y.shape[1], y.shape[2], 1)
    x_test = get_image(os.path.join(data_dir, "Test_Data"))
    y_test = get_image(os.path.join(data_dir, "Test_GroundTruth"), mask=True)
    y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], y_test.shape[2], 1)
    # 转tensor
    db_train = tf.data.Dataset.from_tensor_slices((x, y))
    db_train = db_train.map(preprocess).shuffle(shuffle).batch(batch_size=batch_size)
    db_t = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    db_t = db_t.map(preprocess).batch(batch_size=batch_size)
    return db_train, db_t

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
    parser.add_argument("--epochs",
                        type = int,
                        default = 16,
                        help = "number of epochs",
                        )
    parser.add_argument("--lr",
                        type = float,
                        default = 0.0001,
                        help = "learning rate",
                        )
    parser.add_argument("--momentum",
                        type = float,
                        default = 0.9,
                        help = "momentum",
                        )
    parser.add_argument("--logs",
                        type = str,
                        default = "./logs",
                        help = "日志文件夹",
                        )
    args = parser.parse_args()

    model = Unet()
    model.build(input_shape=(None, 256, 256, 3))
    model.compile(optimizer=optimizers.Adam(lr=args.lr), loss=dice_coef_loss, metrics=['accuracy', dice_coef])

    # 加载数据集
    db, db_test = get_db(args.data_dir, args.batch_size)
    # 设置tf.keras.callbacks.ModelCheckpoint回调实现自动保存模型
    checkpoint_path = "weight/ep{epoch:03d}-val_loss{val_loss:.3f}-val_acc{val_accuracy:.3f}"
    modelCheckpoint = callbacks.ModelCheckpoint(
        filepath=checkpoint_path, # 保存模型的路径
        verbose=1, # 是否输出信息1是0否
        # save_weights_only=True,
        period=1, # 隔几轮保存一次模型
    )
    earlyStopping = callbacks.EarlyStopping(
    monitor='val_loss',  # 被监测的数据
    min_delta=0.001, 
    patience=4, # 能接受提升小于min_delta的轮数
    )
    # 连续patience轮monotor没有提升将改变学习率
    reduceLROnPlateau = callbacks.ReduceLROnPlateau(
        factor=0.2, # new_lr = lr * factor
        patience=3, # 能接受无提升轮数
        min_lr=0.0000001) # lr下限
    tensorboard = callbacks.TensorBoard(
        log_dir=args.logs, 
        write_graph=True, # 在TensorBoard中可视化图像
        update_freq='epoch'# 每个epoch后将损失和指标写入TensorBoard
    )
    # 训练模型
    model.fit(
        db,
        epochs=args.epochs,
        validation_data=db_test,
        validation_freq=1, # 隔几轮测试
        callbacks=[modelCheckpoint, tensorboard, reduceLROnPlateau, earlyStopping],
        batch_size=args.batch_size,
        workers=args.workers,
    )