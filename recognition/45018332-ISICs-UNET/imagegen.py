import tensorflow as tf
from tensorflow import keras as kr
from tensorflow.keras import preprocessing as krp

def create_train_generator(data_path):
    folders = ['/train_img', '/train_mask']

    img_gen = krp.ImageDataGenerator(rescale=1/255)

    new_size = (256,256)
    train_img_gen = img_gen.flow_from_directory((data_path+folders[0]), target_size=new_size, color_mode="grayscale", batch_size=16, class_mode = None, seed=69)
    train_mask_gen = img_gen.flow_from_directory((data_path+folders[1]), target_size=new_size, color_mode="grayscale", batch_size=16, class_mode = None, seed=69)

    train_generator = zip(train_img_gen,train_mask_gen)

def create_val_generator(data_path):
    folders = ['/val_img', '/val_mask']
    img_gen = krp.ImageDataGenerator(rescale=1/255)

    new_size = (256,256)
    val_img_gen = img_gen.flow_from_directory((data_path+folders[0]), target_size=new_size, color_mode="grayscale", batch_size=16, class_mode = None, seed=69)
    val_mask_gen = img_gen.flow_from_directory((data_path+folders[1]), target_size=new_size, color_mode="grayscale", batch_size=16, class_mode = None, seed=69)

    val_generator = zip(val_img_gen,val_mask_gen)

def create_test_generator(data_path):
    folders=['/test_img', '/test_mask']
    img_gen = krp.ImageDataGenerator(rescale=1/255)

    new_size = (256,256)
    test_img_gen = img_gen.flow_from_directory((data_path+folders[0]), target_size=new_size, color_mode="grayscale", batch_size=16, class_mode = None, seed=69)
    test_mask_gen = img_gen.flow_from_directory((data_path+folders[1]), target_size=new_size, color_mode="grayscale", batch_size=16, class_mode = None, seed=69)

    test_generator = zip(test_img_gen,test_mask_gen)