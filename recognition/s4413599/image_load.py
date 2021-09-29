'''
Author: Anqi Yan S4413599
'''

import pathlib
import tensorflow as tf

'''
    Function used to load the image
'''
def load_image(inputPath, seg = False):
    '''
    :param inputPath: The path of the Image File
    :param seg: Set if the images are segmentation image or not
    :return: loaded image
    '''
    train_data_root = pathlib.Path(inputPath)
    train_image_paths = sorted(list(train_data_root.glob('*')))
    train_image_paths = [str(path) for path in train_image_paths]
    train_ds = []
    for img in train_image_paths:
        img_tensor = tf.image.decode_image(tf.io.read_file(img))
        if seg == False:
            img_tensor = tf.image.rgb_to_grayscale(img_tensor, name=None)
        img_tensor = tf.image.resize(img_tensor, [128, 128])
        img_final = img_tensor / 255.0
        train_ds.append(img_final)
    print("Data load successfully")
    return tf.convert_to_tensor(train_ds)

# train_ds= load_image('/Users/anqiyan/Desktop/COMP3710Report/Training_Data_JPG/JPG')
# seg_ds = load_image('/Users/anqiyan/Desktop/COMP3710Report/Training_Data_Seg/SEG', seg=True)








