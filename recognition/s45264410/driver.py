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
from keras.layers import Input, merge, UpSampling2D,BatchNormalization, Add
from keras.models import Model
import tensorflow_addons as tfa

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
  
  def context_block(self,input_mat,num_filters):
      
    
    c1 = Conv2D(num_filters, kernel_size =(3,3), padding = 'same')(input_mat)
    c1 = BatchNormalization()(c1)
    c1 = LeakyReLU(alpha=0.01)(c1)
    c1 = Dropout(.3)(c1)
    
    c2 = Conv2D(num_filters, kernel_size =(3,3), padding = 'same')(c1)
    c2 = BatchNormalization()(c2)
    c2 = LeakyReLU(alpha=0.01)(c2)
    
    # conMod = tfa.layers.InstanceNormalization()(input_mat)
    # conMod = Activation(activation=LeakyReLU(alpha=0.01))(conMod)
    # conMod = Conv2D(num_filters, (3, 3), padding="same")(conMod)
    # conMod = Dropout(0.3)(conMod)

    # conMod = tfa.layers.InstanceNormalization()(conMod)
    # conMod = Activation(activation=LeakyReLU(alpha=0.01))(conMod)
    # conMod = Conv2D(num_filters, (3, 3), padding="same")(conMod)

    return c2

  def upsample_block(self,input_mat,num_filters):
    c1 = UpSampling2D()(input_mat)
    c2 = Conv2D(num_filters, (3,3), activation=LeakyReLU(alpha=0.01), padding = 'same')(c1)
    
    return c2
  
  def local_block(self,input_mat,num_filters):
        # temp = input_mat
        x1 = Conv2D(num_filters, kernel_size =(3,3), activation=LeakyReLU(alpha=0.01), padding = 'same')(input_mat)
        
        x2 = Conv2D(num_filters, kernel_size =(1,1), activation=LeakyReLU(alpha=0.01), padding = 'same')(x1)
        
        return x2
    
  def Unet(self, input_img, n_filters = 16, dropout = 0.2):

    c1 = self.conv_block(input_img,n_filters)
    p1 = MaxPooling2D(pool_size=(2, 2), strides=2)(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = self.conv_block(p1,n_filters*2);
    p2 = MaxPooling2D(pool_size=(2,2) ,strides=2)(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = self.conv_block(p2,n_filters*4);
    p3 = MaxPooling2D(pool_size=(2,2) ,strides=2)(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = self.conv_block(p3,n_filters*8);
    p4 = MaxPooling2D(pool_size=(2,2) ,strides=2)(c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = self.conv_block(p4,n_filters*16);

    u6 = Conv2DTranspose(n_filters*8, (3,3), strides=(2, 2), padding='same')(c5);
    u6 = concatenate([u6,c4]);
    c6 = self.conv_block(u6,n_filters*8)
    c6 = Dropout(dropout)(c6)
    u7 = Conv2DTranspose(n_filters*4,(3,3),strides = (2,2) , padding= 'same')(c6);

    u7 = concatenate([u7,c3]);
    c7 = self.conv_block(u7,n_filters*4)
    c7 = Dropout(dropout)(c7)
    u8 = Conv2DTranspose(n_filters*2,(3,3),strides = (2,2) , padding='same')(c7);
    u8 = concatenate([u8,c2]);

    c8 = self.conv_block(u8,n_filters*2)
    c8 = Dropout(dropout)(c8)
    u9 = Conv2DTranspose(n_filters,(3,3),strides = (2,2) , padding='same')(c8);

    u9 = concatenate([u9,c1]);

    c9 = self.conv_block(u9,n_filters)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    self.model = tf.keras.Model(inputs=input_img, outputs=outputs)
    
  def improved_unet(self, input_img, n_filters = 16, dropout = 0.3):
      
      ## block 1
      c1 = Conv2D(n_filters, (3,3), padding='same')(input_img)
      c1 = LeakyReLU(alpha=0.01)(c1)
      c2 = self.context_block(c1, n_filters)
      add1 = Add()([c1,c2])

      ## block 2
      c3 = Conv2D(n_filters*2, (3,3),padding='same')(add1)
      c3 = LeakyReLU(alpha=0.01)(c3)
      c4 = self.context_block(c3, n_filters*2)
      add2 = Add()([c3,c4])
      
      ## block 3
      c5 = Conv2D(n_filters*4, (3,3),padding='same')(add2)
      c5 = LeakyReLU(alpha=0.01)(c5)
      c6 = self.context_block(c5, n_filters*4)
      add3 = Add()([c5,c6])
      print("Block3 ", add3.shape)
      
      ## block 4
      c7 = Conv2D(n_filters*8, (3,3),padding='same')(add3)
      c7 = LeakyReLU(alpha=0.01)(c7)
      c8 = self.context_block(c7, n_filters*8)
      add4 = Add()([c7,c8])
      print("Block4: ", add4.shape)
      
      ## block 5
      c9 = Conv2D(n_filters*16, kernel_size=(3,3),padding='same')(add4)
      c9 = LeakyReLU(alpha=0.01)(c9)
      c10 = self.context_block(c9, n_filters*16)
      add5 = Add()([c9,c10])
      print("Add1: ", add5.shape)
      
      up1 = self.upsample_block(add5, n_filters*8)
      con1 = concatenate([add4, up1])
      
      ## block 6
      c12 = self.local_block(con1, n_filters*8)
      c12 = self.upsample_block(c12, n_filters*4)
      con2 = concatenate([add3, c12])
      
      ##block 7
      c13 = self.local_block(con2, n_filters*4)
      segment1 = Conv2D(1,(1,1), activation=LeakyReLU(alpha=0.01), padding='same')(c13)
      segment1 = UpSampling2D(interpolation='bilinear')(segment1)
      c13 = self.upsample_block(c13, n_filters*2)
      block7 = concatenate([add2, c13])
      
      ## block 8
      c14 = self.local_block(block7, n_filters*4)
      segment2 = Conv2D(1,(1,1), activation=LeakyReLU(alpha=0.01), padding='same')(c14)
      segment2 = UpSampling2D(interpolation='bilinear')(segment2)
      c14 = self.upsample_block(c14, n_filters)
      block8 = concatenate([add1, c14])
      
      ##block 9
      c15 = Conv2D(n_filters*2, kernel_size=(3,3), padding='same')(block8)
      segment3 = Conv2D(1,(1,1), activation=LeakyReLU(alpha=0.01), padding='same')(c15)
      block9 = Add()([segment1,segment2,segment3])
      
      ## output
      outputs = Conv2D(1, (1, 1), activation='sigmoid')(block9)
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
          plt.title("Base Image")

          # show true mask
          plt.subplot(3, 3, 3*i+2)
          plt.imshow(mask_batch[i])
          plt.axis('off')
          plt.title("True Mask")

          # show predicted mask
          plt.subplot(3, 3, 3*i+3)
          plt.imshow(predictions[i])
          plt.axis('off')
          plt.title("Predicted mask")
      plt.show()    