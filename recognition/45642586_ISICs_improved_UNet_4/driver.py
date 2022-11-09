"""
Improved_UNet for ISIC2018 data set.

COMP3710 Project:
    Question 4: Segment the ISICs data set with the Improved UNet [1] with all labels having a minimum Dice similarity
                coefficient of 0.8 on the test set. [Normal Difficulty]


@author: Xiao Sun
@Student Id: 45642586
"""


import os
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras import layers, models, Input, Model
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
import UNet_model

# "__main__" is at the end.

# load and process data
def load_data():
    
    
    # we use img_input[1:-1] because the first and last file is not image document.
    #load input images and process into tf dataset.
    img_input = os.listdir(r'C:\Users\s4564258\Downloads\ISIC2018_Task1-2_Training_Input_x2')
    img_input = [os.path.join(r'C:\Users\s4564258\Downloads\ISIC2018_Task1-2_Training_Input_x2', path) for path in img_input[1:-1]]
    path_img_input = tf.data.Dataset.from_tensor_slices(img_input)
    image_input_ds = path_img_input.map(data_processing_norm_input, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    
    #load mask images and process into tf dataset.
    img_GroundTruth = os.listdir(r'C:\Users\s4564258\Downloads\ISIC2018_Task1_Training_GroundTruth_x2')
    img_GroundTruth = [os.path.join(r'C:\Users\s4564258\Downloads\ISIC2018_Task1_Training_GroundTruth_x2', path) for path in img_GroundTruth[1:-1]]
    path_img_GroundTruth = tf.data.Dataset.from_tensor_slices(img_GroundTruth)
    image_mask_ds = path_img_GroundTruth.map(data_processing_norm_GT, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    
    image_ds = tf.data.Dataset.zip((image_input_ds, image_mask_ds))
    
    # implot_show(image_input_ds.take(4))
    # implot_show(image_ds.take(4))
    
    return image_ds
    
def data_processing_norm_input(image):
    # process input img data into tf tensor, and normalization.
    
    img_raw = tf.io.read_file(image)
    image = tf.image.decode_jpeg(img_raw, channels=3)
    image = tf.image.resize(image, [256, 256])
    image /= 255.0  # normalize to [0,1] range
    
    return image
    
    
def data_processing_norm_GT(image):
    # process mask (GroundTruth) img data into tf tensor, and normalization.
    
    img_raw = tf.io.read_file(image)
    image = tf.image.decode_jpeg(img_raw, channels=1)
    image = tf.image.resize(image, [256, 256])
    image /= 255.0  # normalize to [0,1] range
    
    return image
    
def implot_show(ds):
    # using imshow to vertify correctly load and process data
    
    for input_img, mask_img in ds:
        display_list = [input_img, mask_img]
        plt.figure(figsize=(18, 18))
        for i in range(len(display_list)):
            print(display_list[i].shape)
            plt.subplot(1, len(display_list), i+1)
            plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
            plt.axis('off')
            plt.show()
      

def split_train_test_val(image_ds):
    # split the whole tf data set into train, validation and test.
    
    # this step will slow down the process.
    # size = len(list(image_ds))
    size = 2594
    
    train_size = int(0.7 * size)
    val_size = int(0.15 * size)
    test_size = int(0.15 * size)
    
    train_image = image_ds.take(train_size)
    val_image = image_ds.skip(train_size)
    test_image = val_image.take(test_size)
    val_image = val_image.skip(test_size)
    

    return train_image, val_image, test_image

# layers
# Build networks
# Details in UNet_model.py module

# train
# Dice coef
def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

# plot dice_coef and loss

def plot_loss(model_history):
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    epochs = range(EPOCHS)

    plt.figure()
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.ylim([0, 1])
    plt.legend()
    plt.show()

def plot_DSC(model_history):
    dsc = model_history.history['dice_coef']
    val_dsc = model_history.history['val_dice_coef']

    epochs = range(EPOCHS)

    plt.figure()
    plt.plot(epochs, dsc, 'r', label='Training DSC')
    plt.plot(epochs, val_dsc, 'b', label='Validation DSC')
    plt.title('Training and Validation DSC')
    plt.xlabel('Epoch')
    plt.ylabel('DSC Value')
    plt.ylim([0, 1])
    plt.legend()
    plt.show()



# plot input/ground_truth/predict image
def implot_show_predict(ds):
    # using imshow to vertify correctly load and process data
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for input_img, mask_img in ds:
        input_img_pred = tf.expand_dims(input_img, axis=0)
        pred_mask = improved_unet_model.predict(input_img_pred)
        display_list = [input_img, mask_img, pred_mask[0]]
        plt.figure(figsize=(12, 12))
        for i in range(len(display_list)):
            plt.subplot(1, 3, i+1)
            plt.title(title[i])
            plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
            plt.axis('off')
        plt.show()
      


if __name__ == "__main__":
    print("Tensorflow version:", tf.__version__)

    # parameters
    EPOCHS = 300
    BATCH_SIZE = 32
    STEPS_PER_EPOCH =1815//BATCH_SIZE

    # load and process data
    image_ds = load_data()
    image_train_all, image_val_all, image_test_all = split_train_test_val(image_ds)

    # set batch size
    image_train = image_train_all.batch(BATCH_SIZE).repeat()
    image_val = image_val_all.batch(BATCH_SIZE)
    image_test = image_test_all.batch(BATCH_SIZE)


    # Improved Unet model
    improved_unet_model = UNet_model.Improved_UNet_model()

    # train
    # learning rate decay
    initial_learning_rate = 0.0005
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1000,
        decay_rate=0.985,
        staircase=True)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)


    improved_unet_model.compile(optimizer=opt, loss=dice_coef_loss, metrics=[dice_coef])

    print("Training on UNet_model...")
    model_history = improved_unet_model.fit(image_train,steps_per_epoch=STEPS_PER_EPOCH ,epochs=EPOCHS, validation_data=image_val)
    print("Finish Training")

    # test
    print("Evaluating on test image...")
    improved_unet_model.evaluate(image_test)
    print("Finish Evaluation")

    # plot input/ground_truth/predict image
    implot_show_predict(image_test_all.take(10))

    print("END")
