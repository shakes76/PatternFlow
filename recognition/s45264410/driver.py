import os
from threading import main_thread
import tensorflow as tf
from absl import flags
import cv2
import glob
import matplotlib.pyplot as plt
from functools import partial
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, LeakyReLU
from keras.layers import Input, merge, UpSampling2D,BatchNormalization
from keras.models import Model

image_path = r'C:\Users\desmo\Downloads\ISIC2018_Task1-2_Training_Data\ISIC2018_Task1-2_Training_Input_x2'
mask_path = r'C:\Users\desmo\Downloads\ISIC2018_Task1-2_Training_Data\ISIC2018_Task1_Training_GroundTruth_x2'

IMG_WIDTH = 256
IMG_HEIGHT = 192
IMG_CHANNELS = 3

MASK_WIDTH = 256
MASK_HEIGHT = 192
MASK_CHANNELS = 1

class pre_process:
  def __init__(self):
    self.train_ds = None
    self.val_ds = None
    self.test_ds = None
    self.model = None
    
  @staticmethod
  def dice_coefficient(y_true, y_pred, smooth=1):
      """
      Calculate Dice similarity coeefficient for use as a metric
      Interpreted for use with same sized masks as:
        2*(number of pixels with same class in both masks, ie the union)
        /2*(number of pixels in each mask)
      Code for this function written by Karan Jakhar (2019). Retrieved from:
      https://medium.com/@karan_jakhar/100-days-of-code-day-7-84e4918cb72c
      """
      y_true = tf.keras.backend.flatten(y_true)
      y_pred = tf.keras.backend.flatten(y_pred)
      intersection = tf.keras.backend.sum(y_true * y_pred)
      return (2.*intersection+smooth) / (tf.keras.backend.sum(y_true)
                                        + tf.keras.backend.sum(y_pred)
                                        + smooth)

  @staticmethod
  def dice_loss(ytrue, y_pred):
      """
      Calculate dice distance for use as a loss function
      Interpreted as what proportion of the predictied mask does not match the
        true mask.
      """
      return 1 - pre_process.dice_coefficient(ytrue, y_pred)  
      
  @staticmethod
  def map_fn(image, mask):
      """
      Helper function to map dataset filenames to the actual image data arrays
      Based on code from COMP3710-demo-code.ipynb from Guest Lecture.
      """

      # load image
      img = tf.io.read_file(image)
      img = tf.image.decode_jpeg(img, channels=IMG_CHANNELS)
      img = tf.image.convert_image_dtype(img, tf.uint8)
      img = tf.image.resize(img, (IMG_WIDTH, IMG_HEIGHT))  # resize all images to min size

      # normalize image to [0,1]
      img = tf.cast(img, tf.float32) / 255.0

      # load mask
      m = tf.io.read_file(mask)
      m = tf.image.decode_png(m, channels=MASK_CHANNELS)
      m = tf.image.resize(m, (MASK_WIDTH, MASK_HEIGHT))  # resize all masks to min size

      # normalize mask to [0,1]
      m = (tf.cast(m, tf.float32) / 255.0)
      
      return img, m
  
  def visualise_loaded_data(self):
        """
        Helper function to visualise loaded image and mask data for sanity checking
        Based on code from COMP3710-demo-code.ipynb from Guest Lecture.
        """
        # fetch some loaded images and true masks
        image_batch, mask_batch = next(iter(self.train_ds.batch(3)))
        
        # visualise images and true masks
        
        plt.figure(figsize=(10, 10))
        for i in range(3):
            plt.subplot(3, 2, 2 * i + 1)
            plt.imshow(image_batch[i])
            plt.title("Input Image")
            plt.axis('off')
            plt.subplot(3, 2, 2 * i + 2)
            plt.imshow(mask_batch[i])
            plt.title("True mask")
            plt.axis('off')
        plt.show()
        
  def get_filepaths(self, directory):
        """
        This function will generate the file names in a directory 
        tree by walking the tree either top-down or bottom-up. For each 
        directory in the tree rooted at directory top (including top itself), 
        it yields a 3-tuple (dirpath, dirnames, filenames).
        """
        file_paths = []  # List which will store all of the full filepaths.

        # Walk the tree.
        for root, directories, files in os.walk(directory):
            for filename in files:
                # Join the two strings in order to form the full filepath.
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)  # Add it to the list.

        return file_paths  # Self-explanatory.    
           
  def load_data(self):
        """
        Downloads and prepares the data set for use in the model
        Based on code from https://www.tensorflow.org/tutorials/load_data/images
        and code from COMP3710-demo-code.ipynb from Guest Lecture.
        """

        # download data
        # data_dir = r'C:\Users\desmo\Downloads\ISIC2018_Task1-2_Training_Data'
        # print("Data dir:", data_dir)

        image_filenames = self.get_filepaths(image_path)
        mask_filenames = self.get_filepaths(mask_path)
        # mask_filenames = [f.replace('.jpg', '_segmentation.png') for f in mask_path]

        # expected number of images is 2594
        image_count = len(image_filenames)
        mask_count = len(mask_filenames)
        print("Image count:", image_count, "Mask count:", mask_count)
        # # split the dataset, 60% train 20% validate 20% test
        val_size = int(image_count * 0.2)
        test_images = image_filenames[:val_size]
        test_masks = mask_filenames[:val_size]
        val_images = image_filenames[val_size:val_size*2]
        val_masks = mask_filenames[val_size:val_size*2]
        train_images = image_filenames[val_size*2:]
        train_masks = mask_filenames[val_size*2:]
        print("Size of training set:", len(train_images), len(train_masks))
        print("Size of validation set:", len(val_images), len(val_masks))
        print("Size of test set:", len(test_images), len(test_masks))

        # create TensorFlow Datasets and shuffle them
        self.train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_masks))
        self.val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_masks))
        self.test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_masks))

        self.train_ds = self.train_ds.shuffle(len(train_images))
        self.val_ds = self.val_ds.shuffle(len(val_images))
        self.test_ds = self.test_ds.shuffle(len(test_images))

        # map filenames to data arrays
        self.train_ds = self.train_ds.map(pre_process.map_fn)
        self.val_ds = self.val_ds.map(pre_process.map_fn)
        self.test_ds = self.test_ds.map(pre_process.map_fn)

        for image, mask in self.train_ds.take(1):
            print('Image shape:', image.numpy().shape)
            print('Mask shape:', mask.numpy().shape)
  

  def Improved_unet(self):
        dropout = 0.1
        # encoder/downsampling
        input_size = (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
        inputs = Input(input_size)
        inputs = tf.keras.Input(input_size)
        conv1 = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation=partial(tf.nn.leaky_relu, alpha=0.01))(inputs)
        conv1 = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation=partial(tf.nn.leaky_relu, alpha=0.01))(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = tf.keras.layers.MaxPool2D((2, 2))(conv1)
        pool1 = tf.keras.layers.Dropout(dropout)(pool1)

        conv2 = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation=partial(tf.nn.leaky_relu, alpha=0.01))(pool1)
        conv2 = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation=partial(tf.nn.leaky_relu, alpha=0.01))(conv2)
        pool2 = tf.keras.layers.MaxPool2D((2, 2))(conv2)
        pool2 = tf.keras.layers.Dropout(dropout)(pool2)
        
        conv3 = tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation=partial(tf.nn.leaky_relu, alpha=0.01))(pool2)
        conv3 = tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation=partial(tf.nn.leaky_relu, alpha=0.01))(conv3)
        conv3 = BatchNormalization()(conv3)
        pool3 = tf.keras.layers.MaxPool2D((2, 2))(conv3)
        pool3 = tf.keras.layers.Dropout(dropout)(pool3)
        
        conv4 = tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation=partial(tf.nn.leaky_relu, alpha=0.01))(pool3)
        conv4 = tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation=partial(tf.nn.leaky_relu, alpha=0.01))(conv4)
        pool4 = tf.keras.layers.MaxPool2D((2, 2))(conv4)
        pool4 = tf.keras.layers.Dropout(dropout)(pool4)
        
        # bridge/bottleneck/shared layer
        conv5 = tf.keras.layers.Conv2D(1024, (3, 3), padding="same", activation=partial(tf.nn.leaky_relu, alpha=0.01))(pool4)
        conv5 = tf.keras.layers.Conv2D(1024, (3, 3), padding="same", activation=partial(tf.nn.leaky_relu, alpha=0.01))(conv5)

        # decoder/upsampling
        up6 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv5)
        up6 = tf.keras.layers.Conv2D(512, (2, 2), padding="same")(up6)
        up6 = tf.keras.layers.concatenate([conv4, up6])
        conv6 = tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation=partial(tf.nn.leaky_relu, alpha=0.01))(up6)
        conv6 = tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation=partial(tf.nn.leaky_relu, alpha=0.01))(conv6)
        conv6 = tf.keras.layers.Dropout(dropout)(conv6)

        up7 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv6)
        up7 = tf.keras.layers.Conv2D(256, (2, 2),  padding="same")(up7)
        up7 = tf.keras.layers.concatenate([conv3, up7])
        conv7 = tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation=partial(tf.nn.leaky_relu, alpha=0.01))(up7)
        conv7 = tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation=partial(tf.nn.leaky_relu, alpha=0.01))(conv7)
        conv7 = tf.keras.layers.Dropout(dropout)(conv7)

        up8 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv7)
        up8 = tf.keras.layers.Conv2D(128, (2, 2), padding="same")(up8)
        up8 = tf.keras.layers.concatenate([conv2, up8])
        conv8 = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation=partial(tf.nn.leaky_relu, alpha=0.01))(up8)
        conv8 = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation=partial(tf.nn.leaky_relu, alpha=0.01))(conv8)
        conv8 = tf.keras.layers.Dropout(dropout)(conv8)

        up9 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv8)
        up9 = tf.keras.layers.Conv2D(64, (2, 2), padding="same")(up9)
        up9 = tf.keras.layers.concatenate([conv1, up9])
        conv9 = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation=partial(tf.nn.leaky_relu, alpha=0.01))(up9)
        conv9 = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation=partial(tf.nn.leaky_relu, alpha=0.01))(conv9)
        conv9 = tf.keras.layers.Dropout(dropout)(conv9)

        # segmentation (output) layer
        outputs = tf.keras.layers.Conv2D(1, (1, 1), padding="same", activation='softmax')(conv9)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

  def conv_block(self,input_mat,num_filters,kernel_size,batch_norm):
        X = Conv2D(num_filters,kernel_size=(kernel_size,kernel_size),strides=(1,1),padding='same')(input_mat)
        if batch_norm:
            X = BatchNormalization()(X)
        
        X = Activation(partial(tf.nn.leaky_relu, alpha=0.01))(X)

        X = Conv2D(num_filters,kernel_size=(kernel_size,kernel_size),strides=(1,1),padding='same')(X)
        if batch_norm:
            X = BatchNormalization()(X)
        
        X = Activation(partial(tf.nn.leaky_relu, alpha=0.01))(X)
        
        return X
    
  def Unet(self, input_img, n_filters = 16, dropout = 0.2, batch_norm = True):

    c1 = self.conv_block(input_img,n_filters,3,batch_norm)
    p1 = MaxPooling2D(pool_size=(2, 2), strides=2)(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = self.conv_block(p1,n_filters*2,3,batch_norm);
    p2 = MaxPooling2D(pool_size=(2,2) ,strides=2)(c2)
    p2 = Dropout(dropout)(p2)

    c3 = self.conv_block(p2,n_filters*4,3,batch_norm);
    p3 = MaxPooling2D(pool_size=(2,2) ,strides=2)(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = self.conv_block(p3,n_filters*8,3,batch_norm);
    p4 = MaxPooling2D(pool_size=(2,2) ,strides=2)(c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = self.conv_block(p4,n_filters*16,3,batch_norm);

    u6 = Conv2DTranspose(n_filters*8, (3,3), strides=(2, 2), padding='same')(c5);
    u6 = concatenate([u6,c4]);
    c6 = self.conv_block(u6,n_filters*8,3,batch_norm)
    c6 = Dropout(dropout)(c6)
    u7 = Conv2DTranspose(n_filters*4,(3,3),strides = (2,2) , padding= 'same')(c6);

    u7 = concatenate([u7,c3]);
    c7 = self.conv_block(u7,n_filters*4,3,batch_norm)
    c7 = Dropout(dropout)(c7)
    u8 = Conv2DTranspose(n_filters*2,(3,3),strides = (2,2) , padding='same')(c7);
    u8 = concatenate([u8,c2]);

    c8 = self.conv_block(u8,n_filters*2,3,batch_norm)
    c8 = Dropout(dropout)(c8)
    u9 = Conv2DTranspose(n_filters,(3,3),strides = (2,2) , padding='same')(c8);

    u9 = concatenate([u9,c1]);

    c9 = self.conv_block(u9,n_filters,3,batch_norm)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    self.model = tf.keras.Model(inputs=input_img, outputs=outputs)

  def show_predictions(self):
      """
      Perform prediction on validation set and report performance
      Based on code from COMP3710-demo-code.ipynb from Guest Lecture.
      """

      # generate predicted masks
      image_batch, mask_batch = next(iter(self.val_ds.batch(3)))
      predictions = self.model.predict(image_batch)

      # visualise images and masks
      import matplotlib.pyplot as plt
      plt.figure(figsize=(20, 10))
      for i in range(3):
          # show base image
          plt.subplot(3, 3, 3*i+1)
          plt.imshow(image_batch[i])
          plt.axis('off')

          # show true mask
          plt.subplot(3, 3, 3*i+2)
          plt.imshow(mask_batch[i])
          plt.axis('off')

          # show predicted mask
          plt.subplot(3, 3, 3*i+3)
          plt.imshow(predictions[i])
          plt.axis('off')
      plt.show()    