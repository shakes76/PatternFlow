import tensorflow as tf
from tensorflow import keras as kr
from tensorflow.keras import optimizers as kro
import os
import matplotlib.pyplot as plt

#other modules written for this report
from processdata import rearr_folders
from imagegen import create_train_generator, create_test_generator, create_val_generator, create_test_batch
from unet import model_unet
from dice import dsc, dsc_loss


def main():
    #this is the driver script for this report
    #limit GPU memory growth, failed to run on my gpu without this part
    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    data_path = './ISIC2018_Task1-2_Training_Data'
    img_path = '/ISIC2018_Task1-2_Training_Input_x2'
    mask_path = '/ISIC2018_Task1_Training_GroundTruth_x2'

    #target image size and color channels
    rows = 128
    cols = 128
    channels = 1

    epoch_no = 5
    batch_size = 8
    
    #assuming images (preprocessed ISICs2018 from BB) already downloaded and unzipped as is in the root directory
    #split images into train-test-validation folders
    train_no, val_no, test_no = rearr_folders(data_path,img_path,mask_path)

    #create Keras image generators to use for training
    train_generator = create_train_generator(data_path)
    val_generator = create_val_generator(data_path)
    test_generator = create_test_generator(data_path)

    #UNET training model
    model = model_unet(rows,cols,channels)
    model.summary()
    model.compile(optimizer=kro.Adam(learning_rate=0.00001), loss=dsc_loss, metrics=[dsc])

    training_results = model.fit(train_generator, epochs=epoch_no, steps_per_epoch=(train_no//batch_size),validation_data=val_generator, validation_steps=(val_no//batch_size))
    #plot loss and dsc against epoch
    plt.figure(1, figsize=(20,10))
    plt.plot(range(len(training_results.history['loss'])),training_results.history['loss'], label='loss')
    plt.plot(range(len(training_results.history['dsc'])),training_results.history['dsc'], label='dsc')
    plt.plot(range(len(training_results.history['val_loss'])),training_results.history['val_loss'], label='val_loss')
    plt.plot(range(len(training_results.history['val_dsc'])),training_results.history['val_dsc'], label='val_dsc')
    plt.legend()
    plt.show()

    #get DSC of trained model on testing dataset
    eval_results = model.evaluate(test_generator, steps=(test_no//batch_size))
    print(eval_results)

    #prediction on testing dataset to visualize
    #code is adapted from COMP3710-demo-code.ipynb as shown in lecture
    img_batch, mask_batch = create_test_batch(data_path)

    predictions = model.predict(test_generator, steps=(test_no//batch_size))

    plt.figure(2)
    for i in range(3):
        plt.subplot(3,3,i+1)
        plt.imshow(img_batch[i])
        plt.axis('off')

        plt.subplot(3,3,i+4)
        plt.imshow(mask_batch[i])
        plt.axis('off')

        plt.subplot(3,3,i+7)
        plt.imshow(predictions[i])
        plt.axis('off')
    
    plt.show()

if __name__ == "__main__":
    main()