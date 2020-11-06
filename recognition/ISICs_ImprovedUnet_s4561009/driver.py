'''
Driver script for training and predicting the ISIC 2018 Challenge dataset
using the Improved U-Net network architecture.

@author Vincentius Aditya Sundjaja
@student_number 45610099
@email s4561009@student.uq.edu.au
@course COMP3710 - Pattern Recognition
@date 6 November 2020
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import os

from matplotlib import image
from pathlib import Path
from tensorflow.keras import backend as K
from IPython.display import clear_output
from model import *


####################################################
################ DATA PREPROCESSING ################
####################################################
def decode_img(image):
    """ A function used for decoding the input image

    Args:
        label (string): a string path to the input image

    Returns:
        tensor object: a tensor object of decoded input image
    """    
    image = tf.image.decode_jpeg(image, channels=3)
    # resize the image to the desired size
    image =  tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    # Standardise values to be in the [0, 1] range.
    image = tf.cast(image, tf.float32) / 255.0
    
    return image
    
def decode_label(label):
    """ A function used for decoding the label image

    Args:
        label (string): a string path to the label image

    Returns:
        tensor object: a tensor object of decoded label image
    """     
    label = tf.image.decode_png(label, channels=1)
    # Resize the image to the desired size.
    label =  tf.image.resize(label, [IMG_HEIGHT, IMG_WIDTH])
    
    # Convert image to a binary image
    label = tf.round(label / 255.0)
    label = tf.cast(label, tf.float32)
    return label

def decode_label_with_onehot(label):
    """ A function used for decoding the label image (with one hot encode)

    Args:
        label (string): a string path to the label image

    Returns:
        tensor object: a tensor object of one hot encoded label image
    """    
    label = tf.image.decode_png(label, channels=1)
    # Resize the image to the desired size.
    label =  tf.image.resize(label, [IMG_HEIGHT, IMG_WIDTH])
    
    # One hot encode the label image
    one_hot_map = []

    for clr in [0, 255]:
        class_map = tf.equal(label, clr)
        class_map = tf.reduce_all(class_map,axis=-1)
        one_hot_map.append(class_map)
    
    one_hot_map = tf.stack(one_hot_map, axis=-1)
    one_hot_map = tf.cast(one_hot_map, tf.float32)
    return one_hot_map

def process_data(image, label):
    """ A function to process the Dataset

    Args:
        image (string): a string path to the input image
        label (string): a string path to the label image

    Returns:
        tf.data.Dataset: A tensorflow dataset object consists of
                         processed input and label images
    """    
    image = tf.io.read_file(image)
    image = decode_img(image)
    
    label = tf.io.read_file(label)
    label = decode_label(label)
    # label = decode_label_with_onehot(label)
    
    return image, label   


#####################################################################
################### MODEL TRAINING AND PREDICTING ###################
#####################################################################
def dice_coef(y_true, y_pred, smooth=1.0):
    """ Function to calculate Dice similarity Coefficient between ground truth label image
        and the predicted label image

    Args:
        y_true (numpy.array): a numpy array of ground truth label image data
        y_pred (numpy.array): a numpy array of predicted label image data
        smooth (float, optional): a number to avoid division by 0

    Returns:
        float: The DSC between y_true and y_pred
    """   
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    """ Dice similarity Coefficient loss function between ground truth label image
        and the predicted label image

    Args:
        y_true (numpy.array): a numpy array of ground truth label image data
        y_pred (numpy.array): a numpy array of predicted label image data

    Returns:
        float: The DSC loss between y_true and y_pred
    """ 
    return 1-dice_coef(y_true, y_pred)

def display(image, ground_truth, prediction, result_dir, output_class_num, num):
    """ Function to display/compare the original image, ground truth label, and 
        prediction label.

    Args:
        image (numpy.array): a numpy array of the original image (input image)
        ground_truth (numpy.array): a numpy array of ground truth label image data
        prediction (numpy.array): a numpy array of predicted label image data
        result_dir (string): path to the result images directory
        num (integer): number of images to display/compare
    """ 
    plt.figure(figsize=(20, 20))
    colors = ['black', 'green', 'red']
    for i in range(num):
        plt.subplot(4, 3, 3*i+1)
        plt.imshow(image[i])
        title = plt.title('The actual image')
        plt.setp(title, color=colors[0])
        plt.axis('off')
        
        plt.subplot(4, 3, 3*i+2)
        if (output_class_num > 1):
            plt.imshow(tf.argmax(ground_truth[i], axis=2))
        else:
            plt.imshow(ground_truth[i])
        title = plt.title('Ground truth image segmentation')
        plt.setp(title, color=colors[1])
        plt.axis('off')
        
        plt.subplot(4, 3, 3*i+3)
        if (output_class_num > 1):
            plt.imshow(tf.argmax(prediction[i], axis=2))
        else:
            plt.imshow(prediction[i] > 0.5)
        title = plt.title('Prediction image segmentation')
        plt.setp(title, color=colors[2])
        plt.axis('off')

        print("DICE SIMILARITY FOR INPUT {}: {}".format(i, dice_coef(ground_truth[i], prediction[i])))
    plt.savefig(result_dir + "results.png")

def show_predictions(model, processed_test_ds, result_dir, output_class_num, num=3):
    image_test_batch, label_test_batch = next(iter(processed_test_ds.batch(num)))
    prediction = model.predict(image_test_batch)
    display(image_test_batch, label_test_batch, prediction, result_dir, output_class_num, num)

def show_plots(history, result_dir):
    plt.subplot(211)
    plt.title('Dice Similarity Coefficient Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss (DSC)")

    # plot accuracy during training
    plt.subplot(212)
    plt.title('Dice Similarity Coefficient')
    plt.plot(history.history['dice_coef'], label='train')
    plt.plot(history.history['val_dice_coef'], label='val')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (DSC)")

    plt.tight_layout()
    plt.savefig(result_dir + "plots.png")

## GLOBAL VARIABLES
IMG_HEIGHT = 192
IMG_WIDTH = 256

#####################################################
################### MAIN FUNCTION ###################
#####################################################
def main():
    ## GETTING THE INPUT FILES
    isic_input_path = './dataset/ISIC2018_Task1-2_Training_Input_x2/*.jpg'
    isic_groundTruth_path = './dataset/ISIC2018_Task1_Training_GroundTruth_x2/*.png'
    isic_input = sorted(glob.glob(isic_input_path))
    isic_groundTruth = sorted(glob.glob(isic_groundTruth_path))

    result_images_dir = "result_images/"
    os.makedirs(result_images_dir, exist_ok=True)

    ## PARAMETERS
    DATASET_SIZE = len(isic_input)
    BATCH_SIZE = 32
    NUM_OF_EPOCH = 10

    TRAIN_SIZE = int(0.7 * DATASET_SIZE)
    VAL_SIZE = int(0.15 * DATASET_SIZE)
    TEST_SIZE = int(0.15 * DATASET_SIZE)

    ## Splitting up the dataset for training, validation, and testing
    full_ds = tf.data.Dataset.from_tensor_slices((isic_input, isic_groundTruth))
    full_ds = full_ds.shuffle(DATASET_SIZE, reshuffle_each_iteration=False)
    train_ds = full_ds.take(TRAIN_SIZE)
    # skip the dataset already used for training
    test_ds = full_ds.skip(TRAIN_SIZE)
    # get the dataset for validation
    val_ds = full_ds.skip(VAL_SIZE)
    # get the dataset for testing
    test_ds = full_ds.take(TEST_SIZE)

    ## Process the raw data
    ## Use Dataset.map to apply this transformation.
    processed_train_ds = train_ds.map(process_data)
    processed_val_ds = val_ds.map(process_data)
    processed_test_ds = test_ds.map(process_data)

    # Getting the input and output size
    input_size = (0, 0, 0)
    output_class_num = 0
    for image, label in processed_train_ds.take(1):
        input_size = image.numpy().shape
        output_class_num = label.numpy().shape[2]

    ## Uncomment below to use the original unet model
    # model = unet(output_class_num, input_size)
    ## Uncomment below to use the improved unet model
    model = improved_unet(output_class_num, input_size)

    ## Using dice similarity coefficient as the loss function and one of the metric
    print("Loss Function: dice similarity coefficient\n")
    model.compile(optimizer='adam', loss=dice_coef_loss, metrics=['accuracy', dice_coef])
    model.summary()

    ## TRAIN THE MODEL
    history = model.fit(processed_train_ds.batch(BATCH_SIZE), 
                        validation_data=processed_val_ds.batch(BATCH_SIZE), 
                        epochs=NUM_OF_EPOCH)
    show_plots(history, result_images_dir)
    print()
    print("FINISHED TRAINING...")
    print()
    
    ## Show some predictions result
    show_predictions(model, processed_test_ds, result_images_dir, output_class_num)
    print()

    ## EVALUATE THE TRAINED MODEL
    [loss, accuracy, dsc] = model.evaluate(processed_test_ds.batch(BATCH_SIZE), verbose=1)
    print("RESULTS FROM THE EVALUATE FUNCTION:")
    print("Loss (dice similarity coefficient):", loss)
    print("Dice Similarity Coefficient:", dsc)
    print("Accuracy (metrics):", accuracy)
    print()

    ## PREDICT ALL THE TEST SET
    image_test_batch, label_test_batch = next(iter(processed_test_ds.batch(TEST_SIZE)))
    predictions = model.predict(image_test_batch)

    ## CALCULATING THE AVERAGE DSC OF ALL PREDICTED TEST IMAGE
    bad_dsc = 0
    total_dsc = 0
    length = predictions.shape[0]
    min_dsc = 0.8
    print("DSC BELOW {}:".format(min_dsc))
    for i in range(length):
        dsc = dice_coef(label_test_batch[i], predictions[i])
        if dsc < min_dsc:
            bad_dsc += 1
            print("  Index {}, dsc is {}".format(i, dsc))
        total_dsc += dsc

    print()
    print("There are {} bad dsc (< 0.8) out of {}".format(bad_dsc, length))
    print("There are {} good dsc (>= 0.8) out of {}".format((length-bad_dsc), length))
    print("Average dsc: ", total_dsc/length)


if __name__ == "__main__":
    main()