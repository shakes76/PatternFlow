'''
Author: Anqi Yan S4413599
'''

import pathlib
import tensorflow as tf
import tqdm
from sklearn.model_selection import train_test_split

'''
    Function used to load the image
'''
def load_image(inputPath, seg = False):
    train_data_root = pathlib.Path(inputPath)
    train_image_paths = sorted(list(train_data_root.glob('*')))
    train_image_paths = [str(path) for path in train_image_paths]
    train_ds = []
    for img in tqdm.tqdm(train_image_paths):
        img_tensor = tf.image.decode_image(tf.io.read_file(img))
        if seg == False:
            img_tensor = tf.image.rgb_to_grayscale(img_tensor, name=None)
        img_tensor = tf.image.resize(img_tensor, [128, 128])
        if seg == True:
            img_final = img_tensor // 255.0
            img_final = tf.cast(img_final,tf.int8)
        else:
            img_final = img_tensor / 255.0
        train_ds.append(img_final)
    print("Data load successfully")
    return tf.convert_to_tensor(train_ds)

def get_train_test_data(train_img = '/Users/anqiyan/Desktop/COMP3710Report/Training_Data_JPG/JPG', seg_img = '/Users/anqiyan/Desktop/COMP3710Report/Training_Data_Seg/SEG'):
    '''
    :param train_img: path to the training image
    :param seg_img: path to the segmentation image
    :return: splited imaeg
    '''
    train_ds= load_image(train_img)
    seg_ds = load_image(seg_img, seg=True)
    X_train, X_test, y_train, y_test = train_test_split(train_ds.numpy(), seg_ds.numpy(), test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test









