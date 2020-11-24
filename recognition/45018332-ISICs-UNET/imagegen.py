import tensorflow as tf
from tensorflow import keras as kr
from tensorflow.keras import preprocessing as krp

#These functions create Keras image generators that loads data in batches to be used for training
#Training, validation and testing datasets are loaded separately
def create_train_generator(data_path):
    folders = ['/train_img', '/train_mask']

    img_gen = krp.image.ImageDataGenerator(rescale=1/255)

    new_size = (128,128)
    train_img_gen = img_gen.flow_from_directory((data_path+folders[0]), target_size=new_size, color_mode="grayscale", batch_size=8, class_mode = None, seed=69)
    train_mask_gen = img_gen.flow_from_directory((data_path+folders[1]), target_size=new_size, color_mode="grayscale", batch_size=8, class_mode = None, seed=69)

    return zip(train_img_gen,train_mask_gen)

def create_val_generator(data_path):
    folders = ['/val_img', '/val_mask']
    img_gen = krp.image.ImageDataGenerator(rescale=1/255)

    new_size = (128,128)
    val_img_gen = img_gen.flow_from_directory((data_path+folders[0]), target_size=new_size, color_mode="grayscale", batch_size=8, class_mode = None, seed=69)
    val_mask_gen = img_gen.flow_from_directory((data_path+folders[1]), target_size=new_size, color_mode="grayscale", batch_size=8, class_mode = None, seed=69)

    return zip(val_img_gen,val_mask_gen)

def create_test_generator(data_path):
    folders=['/test_img', '/test_mask']
    img_gen = krp.image.ImageDataGenerator(rescale=1/255)

    new_size = (128,128)
    test_img_gen = img_gen.flow_from_directory((data_path+folders[0]), target_size=new_size, color_mode="grayscale", batch_size=8, class_mode = None, seed=69)
    test_mask_gen = img_gen.flow_from_directory((data_path+folders[1]), target_size=new_size, color_mode="grayscale", batch_size=8, class_mode = None, seed=69)

    return zip(test_img_gen,test_mask_gen)

def create_test_batch(data_path):
    folders=['/test_img', '/test_mask']
    img_gen = krp.image.ImageDataGenerator(rescale=1/255)

    new_size = (128,128)
    test_img_gen = img_gen.flow_from_directory((data_path+folders[0]), target_size=new_size, color_mode="grayscale", batch_size=8, class_mode = None, seed=69)
    test_mask_gen = img_gen.flow_from_directory((data_path+folders[1]), target_size=new_size, color_mode="grayscale", batch_size=8, class_mode = None, seed=69)

    img_batch = next(test_img_gen)
    mask_batch = next(test_mask_gen)
    return img_batch, mask_batch