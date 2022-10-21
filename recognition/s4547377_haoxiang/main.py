from matplotlib import image
from pathlib import Path
from PIL import Image
from numpy import asarray
from tensorflow.keras import backend
from IPython.display import clear_output
from model.py import *
import glob
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

'''
Define the paths to the image sets
'''
training_images_path='H:/githubcomp3710/PatternFlow/recognition/s4547377_haoxiang/ISBI2016_ISIC_Part1_Training_Data/ISBI2016_ISIC_Part1_Training_Data/*.jpg'
groundTruth_path='H:/githubcomp3710/PatternFlow/recognition/s4547377_haoxiang/ISBI2016_ISIC_Part1_Training_GroundTruth/ISBI2016_ISIC_Part1_Training_GroundTruth/*.png'

'''
Function explanations:
1.sorted(): It sorts the elements of a given iterable in a specific order (ascending or descending) and returns it as a list.
2.glob(): Return a possibly-empty list of path names that match pathname, 
    which must be a string containing a path specification. 
'''
training_images=sorted(glob.glob(training_images_path))
groundTruth_images=sorted(glob.glob(groundTruth_path))

'''
Define the batch size, image height, image width, image channels and the size of data set
'''
s_dataset=len(training_images)
s_batch=32
height=192
width=256
number_channel=4

'''
Allocate the sizes of three data set: trainging set(0.7), validation set(0.15), test size(0.15)
'''
s_train=int(0.7*s_dataset)
s_validation=int(0.15*s_dataset)
s_test=int(0.15*s_dataset)

#Generate training, validation and test datasets.
'''
Function explanations:
1.from_tensor_slices():Creates a Dataset whose elements are slices of the given tensors.
2.shuffle():Randomly shuffles the elements of this dataset.
3.take():Creates a Dataset with at most count elements from this dataset.
4.skip():Creates a Dataset that skips count elements from this dataset.
'''
complete_dataset=tf.data.Dataset.from_tensor_slices((training_images, groundTruth_images))
complete_dataset=complete_dataset.shuffle(s_dataset, reshuffle_each_iteration=False)
training_dataset=complete_dataset.take(s_train)
test_dataset=complete_dataset.skip(s_train)
validation_dataset=complete_dataset.skip(s_validation)
test_dataset=complete_dataset.take(s_test)

'''
This section aims to pre-process the training images and the ground truth images
Function Explanations:
1.tf.image.decode_jpeg(): Decode a JPEG-encoded image to a uint8 tensor.
2.tf.image.resize(): Resize images to the given size. (second parameter)
3.tf.cast(): Casts a tensor to a new type.
4.tf.image.decode_png: Decode a PNG-encoded image to a uint8 or uint16 tensor.
5.tf.round: Rounds the values of a tensor to the nearest integer, element-wise.
6.tf.equal(): Returns the truth value of (x == y) element-wise.
'''
'''
process_training(): This function takes training images as input, and pre-processes them.
1. Convert to the tensor 
2. Resize it
3. Normalize it
'''
def process_training(inputs):
    #Change the input image into tensor
    inputs=tf.image.decode_jpeg(inputs,channels=3)
    #Resize the image
    inputs=tf.image.resize(inputs,[height,width])
    #Standardise values
    inputs=tf.cast(inputs,tf.float32)/255.0   
    return inputs

'''
process_groundtruth(): This function takes ground truth images as input, and pre-processes them.
1. Convert to the tensor 
(This part is different because I find out that the ground truth images are .png)
2. Resize it
3. Normalize it
'''    
def process_groundtruth(inputs):
    inputs=tf.image.decode_png(inputs,channels=1)
    inputs=tf.image.resize(inputs,[height,width])
    inputs=tf.round(inputs/255.0)
    inputs=tf.cast(inputs,tf.float32)
    return inputs

'''
This function simply uses the function above to process all the image data
'''
def process_images(training, groundtruth):
    training=tf.io.read_file(training)
    training=process_training(training)  
    groundtruth=tf.io.read_file(groundtruth)
    groundtruth=process_groundtruth(groundtruth)    
    return training, groundtruth

'''
Dataset.map():This transformation applies map_func to each element of this dataset, 
and returns a new dataset containing the transformed elements, in the same order as they appeared in the input.
In this case, it is the same as the "apply()" function.
'''
treated_training_set=training_dataset.map(process_images)
treated_validation_set=validation_dataset.map(process_images)
treated_test_set=test_dataset.map(process_images)

'''
This section aims to find the sizes of input and output
'''
input_size=(0, 0, 0)
number_output_classes=0
#Taking only one is enough
for image, label in treated_training_set.take(1):
    input_size=image.numpy().shape
    number_output_classes=label.numpy().shape[2]
plt.figure(figsize=(25, 25))
plt.subplot(1, 4, 1)
plt.imshow(image.numpy())
plt.axis('off')
plt.subplot(1, 4, 2)
if (number_output_classes>1):
    plt.imshow(tf.argmax(label.numpy(),axis=2))
else:
    plt.imshow(label.numpy())
plt.axis('off')

'''
This section aims to create functions for the DSC: Dice similarity coefficient, DSC's images display, predication function 
and the call back function.
'''
'''
Functions Explanations:
1.backend.flatten(): Flatten a tensor.
2.backend.sum(): Sum of the values in a tensor, alongside the specified axis.
'''
#DC lore: The Dice Coefficient is 2 * the Area of Overlap divided by the total number of pixels in both images.
def DSC(y_true, y_pred, smooth=1):
    flatten_y_true=backend.flatten(y_true)
    flatten_y_pred=backend.flatten(y_pred)
    intersection=backend.sum(flatten_y_true*flatten_y_pred)
    return (2.0*intersection+smooth)/(backend.sum(flatten_y_true*flatten_y_true)+backend.sum(flatten_y_pred*flatten_y_pred)+smooth)

#Loss function
def loss_DSC(y_true, y_pred):
    return 1-DSC(y_true, y_pred)

#This function aims to display images
def display_images(image, groundtruth, prediction, number):
    plt.figure(figsize=(20, 20))
    colors = ['maroon', 'navy', 'fuchsia']
    for i in range(number):
        plt.subplot(4, 3, 3*i+1)
        plt.imshow(image[i])
        title = plt.title('Origin Image')
        plt.setp(title, color=colors[0])
        plt.axis('off')
        plt.subplot(4, 3, 3*i+2)
        if (number_output_classes > 1):
            plt.imshow(tf.argmax(groundtruth[i], axis=2))
        else:
            plt.imshow(groundtruth[i])
        title = plt.title('Ground Truth Segmentation')
        plt.setp(title, color=colors[1])
        plt.axis('off')
        plt.subplot(4, 3, 3*i+3)
        if (number_output_classes > 1):
            plt.imshow(tf.argmax(prediction[i], axis=2))
        else:
            plt.imshow(prediction[i] > 0.5)
        title = plt.title('Prediction Segmentation')
        plt.setp(title, color=colors[2])
        plt.axis('off')
        print("Dice similarity {}: {}".format(i, DSC(groundtruth[i], prediction[i])))
    plt.show()

def get_predictions(treated_test_set, num=3):
    batch_image, batch_label = next(iter(treated_test_set.batch(num)))
    prediction = model.predict(batch_image)
    display_images(batch_image, batch_label, prediction, num)

#Call back
'''
tf.keras.callbacks.Callback: Abstract base class used to build new callbacks.
This class is necessary since I need to use it in the training loops.
'''
class DisplayCallback(tf.keras.callbacks.Callback):
    #Called at the end of an epoch. Subclasses should override for any actions to run. 
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)#clear the output of a cell.
        get_predictions(treated_test_set)

#Create improved UNET variable 
model=improved_unet(number_output_classes, input_size)
#Compile the improved UNET model with adam optimizer
model.compile(optimizer='adam', loss=loss_DSC, metrics=['accuracy', DSC])
#Train the model
model_history=model.fit(treated_training_set.batch(s_batch), validation_data=treated_validation_set.batch(s_batch), epochs=60, callbacks=[DisplayCallback()])
#Evaluate
result=model.evaluate(treated_test_set.batch(s_batch), verbose=1)
#Predications
batch_image, batch_label=next(iter(treated_test_set.batch(s_test)))
prediction=model.predict(batch_image)
#Calculate the average DSC value
total_values=0
length=prediction.shape[0]
for i in range(length):
    each_DSC_value=DSC(batch_label[i], prediction[i])
    total_values+=each_DSC_value
print("Average DSC value is ", total_values/length)

#plots
plt.subplot(211)
plt.title('DSC loss Graph')
plt.plot(model_history.history['loss'], label='training_set')
plt.plot(model_history.history['val_loss'], label='validation_set')
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Loss value")
plt.subplot(212)
plt.title('DSC Graph')
plt.plot(model_history.history['DSC'], label='training_set')
plt.plot(model_history.history['val_DSC'], label='validation_set')
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.show()
