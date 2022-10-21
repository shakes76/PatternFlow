import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from IPython.display import display
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img


# return data set(in batch size) from directory where each images belong to one 
# class is in the subdirectory 
"""load data and nomalized"""
def get_train():
    data_train = image_dataset_from_directory(
        "C:/Users/Administrator/Desktop/ANDI/AD_NC/train", labels = "inferred", label_mode= None, image_size=(128, 128), batch_size=32, validation_split=0.3, subset="training", seed=1
    )

    train_dataset = data_train.map(lambda x: x / 255.0)
    print("train data: ", data_train)
    return train_dataset


def get_validation():
    data_val = image_dataset_from_directory(
        "C:/Users/Administrator/Desktop/ANDI/AD_NC/train", labels = "inferred", label_mode= None, image_size=(128, 128), batch_size=32, validation_split=0.3, subset="validation", seed=1
    )

    val_dataset = data_val.map(lambda x: x / 255.0)
    print("val data: ", data_val)
    return val_dataset


def get_test():
    data_test = image_dataset_from_directory(
        "C:/Users/Administrator/Desktop/ANDI/AD_NC/test", labels = "inferred", label_mode= None, image_size=(128, 128), batch_size=32
    )

    test_dataset = data_test.map(lambda x: x / 255.0)
    print("test data: ", data_test)
    return test_dataset



""" rerurn img array representation of one image, return img array reprentation """
def get_img(data):
    # returns a dataset equivalent to iterate over with form of (data, label),
    imgs = data.take(1)
    iterator = iter(imgs)
    img_array = iterator.get_next()[0]

    return img_array

"""down-scale the image and the upscale back by bicubic interpolation"""
def resize_img(img_data, size):
    resized_img_data = tf.image.resize(img_data, [size, size], method='bicubic')
    resized_img_data = tf.image.resize(resized_img_data, (128, 128), method='bicubic')
    return resized_img_data


""" resize the input image, return (crop_img, original)"""
@tf.autograph.experimental.do_not_convert
def resize_img_flow(data, size):
    data = data.map(
        lambda x: (resize_img(x, size), x))
    return data
