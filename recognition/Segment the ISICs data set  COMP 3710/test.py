import os
import cv2
import argparse
import numpy as np
from train import Unet
from tensorflow.keras import backend as K

def get_image(size=(256, 256),data_dir="datasets", img_path="Test_Data", mask_path="Test_GroundTruth"):
    '''
    Obtain the picture
    :param size: the size of the picture
    :param data_dir: the data source
    :return: the picture
    '''
    X = []
    Y = []
    for i in os.listdir(os.path.join(data_dir, img_path)):
        img = cv2.imread(os.path.join(data_dir, img_path, i)) / 255
        img = cv2.resize(img, size)
        X.append(img)
        mask = cv2.imread(os.path.join(data_dir, mask_path, i[:-4]+"_Segmentation.png"), cv2.IMREAD_GRAYSCALE) / 255
        mask[mask>0.5] = 1
        mask[mask<=0.5] = 0
        mask = cv2.resize(mask, size)
        Y.append(mask)
    return np.array(X), np.array(Y)

def get_db(data_dir):
    '''
    Obtain the dataset through specific location
    :param data_dir: specific location
    :return: the dataset
    '''
    X, Y = get_image(data_dir)
    Y = Y.reshape((Y.shape[0], Y.shape[1], Y.shape[2], 1))
    return X, Y

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        type=str,
                        default = "datasets",
                        help = "dataset location",
                        )
    parser.add_argument('--model',
                        default='weight/ep010-val_loss0.456-val_acc0.807',
                        type=str,
                        help='model path')
    args = parser.parse_args()

    model = Unet()
    model.build(input_shape=(None, 256, 256, 3))
    model.load_weights(args.model)
    x_test, y_test = get_db(args.data_dir)
    sum_dice = 0
    i = 0
    # Calculate the Dice similarity coefficient
    for x, y in zip(x_test, y_test):
        y = np.round(y)
        i += 1
        x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
        out = model.predict(x)
        pre = out.ravel()
        target = y.ravel()
        intersection = pre * target
        dice = (2. * np.sum(intersection) + 1) / (np.sum(pre) + np.sum(target) + 1)
        sum_dice += dice
    print(sum_dice / i)

