import os
from threading import main_thread
import tensorflow as tf
from absl import flags
import cv2
import glob
import matplotlib.pyplot as plt
from functools import partial
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, LeakyReLU, Add
from tensorflow.keras.layers import Input, UpSampling2D,BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.python.ops.gen_array_ops import pad
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
  
  def conv_block(self,input_mat,num_filters):
    c1 = tfa.layers.InstanceNormalization()(input_mat)
    c1 = Activation(activation=LeakyReLU(alpha=0.01))(c1)
    c1 = Conv2D(num_filters, (3, 3), padding="same")(c1)
    c1 = Dropout(0.3)(c1)

    c1 = tfa.layers.InstanceNormalization()(c1)
    c1 = Activation(activation=LeakyReLU(alpha=0.01))(c1)
    c1 = Conv2D(num_filters, (3, 3), padding="same")(c1)

    return c1
    
  def upsample_block(self,input_mat,num_filters):
    u1 = UpSampling2D()(input_mat)
    u1 = Conv2D(num_filters, (3,3), activation=LeakyReLU(alpha=0.01), padding='same')(u1)
    
    return u1
  
  def local_block(self, input_mat, num_filter):
    loc1 = Conv2D(num_filter, (3,3), activation=LeakyReLU(alpha=0.01), padding='same')(input_mat)
    loc2 = Conv2D(num_filter, (1,1), activation=LeakyReLU(alpha=0.01), padding='same')(input_mat)
    
    return loc2
    
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

  def Improved_unet(self, input_img, n_filter = 16):
    
    c1 = Conv2D(n_filter * 1, (3,3), activation=LeakyReLU(alpha=0.01), padding='same')(input_img)
    p1 = self.conv_block(c1, n_filter)
    add1 = Add()([p1,c1])
    
    c2 = Conv2D(n_filter * 2, (3,3),strides=2 ,activation=LeakyReLU(alpha=0.01), padding='same')(add1)
    p2 = self.conv_block(c2, n_filter * 2)
    add2 = Add()([p2,c2])

    c3 = Conv2D(n_filter * 4, (3,3),strides=2 ,activation=LeakyReLU(alpha=0.01), padding='same')(add2)
    p3 = self.conv_block(c3, n_filter * 4)
    add3 = Add()([p3,c3])
    
    c4 = Conv2D(n_filter * 8, (3,3),strides=2 ,activation=LeakyReLU(alpha=0.01), padding='same')(add3)
    p4 = self.conv_block(c4, n_filter * 8)
    add4 = Add()([p4,c4])
    
    c5 = Conv2D(n_filter * 16, (3,3),strides=2 ,activation=LeakyReLU(alpha=0.01), padding='same')(add4)
    p5 = self.conv_block(c5, n_filter * 16)
    add5 = Add()([p5,c5])
    
    up1 = self.upsample_block(add5, n_filter * 8)
    con1 = concatenate([up1, add4])
    
    loc1 = self.local_block(con1, n_filter * 8)
    up2 = self.upsample_block(loc1, n_filter * 4)
    con2  = concatenate([up2, add3])
    
    loc2 = self.local_block(con2, n_filter * 4)
    up3 = self.upsample_block(loc2, n_filter * 2)
    con3  = concatenate([up3, add2])
    ## segmentation layer
    segment1 = Conv2D(1, (1,1), activation=LeakyReLU(alpha=0.01), padding='same')(loc2)
    segment1 = UpSampling2D(interpolation='bilinear')(segment1)
    
    loc3 = self.local_block(con3, n_filter * 2)
    up4 = self.upsample_block(loc3, n_filter * 1)
    con4  = concatenate([up4, add1])
    ## segmentation layer
    segment_ = Conv2D(1, (1,1), activation=LeakyReLU(alpha=0.01), padding='same')(loc3)
    segment2 = Add()([segment1, segment_])
    segment2 = UpSampling2D(interpolation='bilinear')(segment2)
    
    conv_final = Conv2D(32, (3,3), activation=LeakyReLU(alpha=0.01), padding='same')(con4)
    segment3 = Conv2D(1,(1,1), activation=LeakyReLU(alpha=0.01),padding='same')(conv_final)
    segment_final = Add()([segment2, segment3])
    
    
    outputs = Conv2D(1, (1,1), activation='sigmoid', padding='same')(segment_final)
    
    self.model = Model(inputs = input_img, outputs= outputs)
    
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
          plt.title("Input image")
          plt.axis('off')

          # show true mask
          plt.subplot(3, 3, 3*i+2)
          plt.imshow(mask_batch[i])
          plt.title("True mask")
          plt.axis('off')

          # show predicted mask
          plt.subplot(3, 3, 3*i+3)
          plt.imshow(predictions[i])
          plt.title("Predicted mask")
          plt.axis('off')
      plt.show()    