# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 08:04:47 2021

@author: jmill
"""

import tensorflow as tf
import glob
import random
from model import ImprovedUnet
import matplotlib.pyplot as plt



#Method for loading the images
def parse_image(img_path):
    image = tf.io.read_file(img_path)
    #load image, these are jpegs
    image = tf.image.decode_jpeg(image, channels = 3)
    image = tf.image.convert_image_dtype(image, tf.uint8)
    
    #Change the image path to the segment folder instead
    mask_path = tf.strings.regex_replace(img_path, "ISIC2018_Task1-2_Training_Input_x2", "ISIC2018_Task1_Training_GroundTruth_x2")
    mask_path = tf.strings.regex_replace(mask_path, ".jpg", "_segmentation.png")
    mask = tf.io.read_file(mask_path)
    #load image, segment masks are png files.
    mask = tf.image.decode_png(mask, channels = 1)
    return {'image': image, 'seg_mask': mask}

def normalize(input_image, input_mask):
    #Normalize the data, tends to segment better when this is done
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask = tf.cast(input_mask, tf.float32) / 255.0
    #Round the input mask to ensure that it is either 1 or 0. There are no extra classes associated with the image segmentation.
    #There are some issues with the masks that are probably due to interpolation when resizing. This removes it and makes the segmentation masks uniform.
    input_mask = tf.round(input_mask)
    return input_image, input_mask


def load_image_train(datapoint: dict):

    #Resize the image to the image_size above.
    input_image = tf.image.resize(datapoint['image'], (image_size, image_size))
    input_mask = tf.image.resize(datapoint['seg_mask'], (image_size, image_size), method = "nearest")
    #Randomly mirror and flip the image to improve training process
    rand_val = 0.25
    if rand_val < 0.25:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)
    elif rand_val < 0.5:
        input_image = tf.image.flip_up_down(input_image)
        input_mask = tf.image.flip_up_down(input_mask)
    elif rand_val < 0.75:
        input_image = tf.image.flip_left_right(input_image)
        input_image = tf.image.flip_up_down(input_image)
        input_mask = tf.image.flip_left_right(input_mask)
        input_mask = tf.image.flip_up_down(input_mask)
    #Normalize image per above method
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask


def load_image_test(datapoint: dict):
    #Load the test data, similar to training method but no modifying the image (except normalization)
    input_image = tf.image.resize(datapoint['image'], (image_size, image_size))
    input_mask = tf.image.resize(datapoint['seg_mask'], (image_size, image_size), method = "nearest")
    input_image, input_mask = normalize(input_image, input_mask)
    
    return input_image, input_mask


def create_mask(pred_mask):
    #Create the mask, by using argmax to get the highest value in the channel and then expanding the dimension to keep the shape correct
     pred_mask = tf.argmax(pred_mask, axis = -1)
     pred_mask = tf.expand_dims(pred_mask, axis = -1)
     return pred_mask


def display_image(image):
    #Create a plot of the appropriate size using cmap = "gist_gray". Best mask to get black/white segmentation.
    plt.figure(figsize = (24,24))
    plt.imshow(tf.keras.preprocessing.image.array_to_img(image), cmap = "gist_gray")
    plt.axis("off")
    plt.show()
    
    
def show_dice_accuracy(dataset):
    #Ensure a dataset has been passed
    if dataset:
        #Score/num_val will be the end result, will add to score for each image and then will divide by num_val
        score = 0
        num_val = 0
      
        for image, mask in dataset:
            #As the images are in batches, this will bulk predict the batch and then we will iterate through the batch individually.
            pred_mask = model.predict(image)
            for i in range(image.shape[0]):
                with tf.device('/device:CPU:0'):    #Remove this if GPU can handle additional memory alloc after training the model
                    num_val = num_val + 1
                    #Alloc True Positive, False positive/False Negative and True Negative values
                    TP = 0
                    FPFN = 0
                    TN = 0
                    
                    #Generate the seg_mask into an image format
                    temp_mask = create_mask(pred_mask[i])
                    
                    #cast the original mask into a int64 (it is a float)
                    new_mask = tf.Variable(mask[i])
                    new_mask = tf.cast(new_mask, tf.int64)
                    #Add the two masks together
                    eval_mask = new_mask + temp_mask
                    #Get counts of each value
                    y,idx,count = tf.unique_with_counts(tf.reshape(eval_mask,-1))
                #The below code works out the TP/FP/FN/TN values
                #This works because TP = 1+1, FP = 1+0, FN = 0+1, TN = 0+0. 
                #As such we can quickly pull each of these values out of the grid and then calculate the dice similarity.
                for j in range(len(y)):
                    if y[j] == 0:
                        TN = count[j].numpy()
                    elif y[j] == 1:
                        FPFN = count[j].numpy()
                    elif y[j] ==2:
                        TP = count[j].numpy()
                #Quick check on the off chance that there are no lesions in the mask (all TN) as we can't divide by a 0.
                if(TP+FPFN == 0):
                    if TN > 0:
                        dice = 1
                else:
                    #Dice similarity value here
                    dice = (2*TP)/(2*TP + FPFN)
               #Sum the score val with the dice. At the end this will be divided and turned into the average.
                score = score + dice
        return (score/num_val)
    else:
        return("Error - No dataset")


if __name__ == "__main__":
    
    
    #Initial variables for running model, change depending on computer power etc.

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    batch_size = 10
    buffer_size = 1000
    #Image size here is lower than the full resolution of the image, can be scaled up depending on GPU Memory
    image_size = 128
    seed = 15
    
    #Load the image file paths and shuffle them
    dataset_path = "D:/Drive D/Data Science/Semester 4/COMP3710/ISIC/ISIC2018_Task1-2_Training_Input_x2/*"
    files = glob.glob(dataset_path)
    random.shuffle(files)     
    train_data, validate_data, test_data = files[:2078], files[2078:2337], files[2337:2596]
    
    #Load the file paths into a arrays, .map loads the image data
    train_set = tf.data.Dataset.list_files(train_data, seed = seed)
    train_set = train_set.map(parse_image)
    test_set = tf.data.Dataset.list_files(test_data, seed = seed)
    test_set = test_set.map(parse_image)
    val_set = tf.data.Dataset.list_files(validate_data, seed = seed)
    val_set = val_set.map(parse_image)
    
    
    #For each set of data, create the tensorflow dataset for each one.
    #Map the correct load_image method, split the dataset into batches to make it easier to run on lower end systems.
    train_dataset = {"train": train_set}
    
    train_dataset['train'] = train_dataset['train'].map(load_image_train, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    train_dataset['train'] = train_dataset['train'].shuffle(buffer_size = buffer_size, seed = seed)
    train_dataset['train'] = train_dataset['train'].repeat()
    train_dataset['train'] = train_dataset['train'].batch(batch_size)
    train_dataset['train'] = train_dataset['train'].prefetch(buffer_size = AUTOTUNE)
    
    val_dataset = {"val": val_set}
    
    val_dataset['val'] = val_dataset['val'].map(load_image_test)
    val_dataset['val'] = val_dataset['val'].repeat()
    val_dataset['val'] = val_dataset['val'].batch(batch_size)
    val_dataset['val'] = val_dataset['val'].prefetch(buffer_size=AUTOTUNE)
    
    test_dataset = {"test": test_set}
    
    test_dataset['test'] = test_dataset['test'].map(load_image_test)
    test_dataset['test'] = test_dataset['test'].batch(batch_size)
    test_dataset['test'] = test_dataset['test'].prefetch(buffer_size=AUTOTUNE)
    
    
    #Create the model from the module. 
    model = ImprovedUnet().create_model(128)
    
    #Split up the batches per epoch
    steps_per_epoch = 2078 // batch_size
    validation_steps = 259 // batch_size
    
    #Train the model
    history = model.fit(train_dataset['train'], epochs = 15, steps_per_epoch = steps_per_epoch, validation_steps = validation_steps, validation_data = val_dataset['val'])
    
    #Model summary
    print(model.summary())
    
    #Print out one of the batches of images. Images always come in form of Image, original mask, generated mask.
    for image, mask in test_dataset['test'].take(1):
        pred_mask = model.predict(image)
        for i in range(image.shape[0]):
            display_image(image[i])
            display_image(mask[i])
            display_image(create_mask(pred_mask[i]))
       
    #Print the test accuracy using the Dice method.
    print("The test accuracy is: " + str(show_dice_accuracy(test_dataset['test'])))
    

    
    
    
    
    
    
    
    
    
    