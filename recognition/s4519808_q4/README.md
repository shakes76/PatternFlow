# Segmentation of ISICs data set using the Improved UNet [[1]](https://arxiv.org/abs/1802.10508v1)

This is the last assignment from Course COMP3710 Pattern Recognition in the University of Queensland in Semester 2, 2020.

# Problem --- Image Segmentation

ISICs stands for International Skin Imaging Collaboration, and the goal of this is to help participants develop image analysis tools to automatically diagnose melanoma from dermoscopic images. 
This project here, is to try to **segment** original skin **RGB images into monochrome images** which represents the possible area of skin lesions, in order to assist cutaneous melanoma diagnosis.

# Algorithm --- Improved UNet 

The improved UNet is developed by F. Isensee, P. Kickingereder, W. Wick, M. Bendszus, and K. H. Maier-Hein. [[1]](https://arxiv.org/abs/1802.10508v1) This deep learning net work is used to handle Brain Tumor Segmentation problem and deal with 3D images in the paper. But here I borrow this structure to cope with my **2D image segmentation problem**. 

![image_unet](./images/unet.png)

Figure above shows the structure of the improved UNet. 
- The *context module* is a pre-activation residual block, with two 3x3 convolutional layers and a dropout layer with p=0.3 in the middle. Noted that, the activation layer uses Leaky ReLU, and batch normalization is changed to instance normalization.
- The *upsampling module* is simply a upsampling2D layer followed by a 3x3 convolution that halves the number of feature map.
- The *localization module* contains a 3x3 convolution and then a 1x1 convolution which halves the number of feature maps.

Here is the code to generate the network.
```python
def  context_module(inputs, filters):
""" filters is the output size of the module"""
	bn1 = tfa.layers.InstanceNormalization()(inputs)
	relu1 = LeakyReLU(alpha=0.01)(bn1)
	conv1 = Conv2D(filters, (3,3), padding='same')(relu1)
	dropout = Dropout(0.3)(conv1)
	bn2 = tfa.layers.InstanceNormalization()(dropout)
	relu2 = LeakyReLU(alpha=0.01)(bn2)
	conv2 = Conv2D(filters, (3,3), padding='same')(relu2)
	return conv2
	
def  upsampling_module(inputs, filters):
""" filters is the output size of the module"""
	up = UpSampling2D(size=(2,2))(inputs)
	conv = Conv2D(filters, (3,3), padding='same')(up)
	return conv
	
def  localization_module(inputs, filters):
""" filters is the output size of the module"""
	conv1 = Conv2D(filters*2, (3,3), padding='same')(inputs)
	conv2 = Conv2D(filters, (1,1))(conv1)
	return conv2
	
# Build the Model
def  improved_unet(h, w):
"""cm, um, lm stand for differnet modules"""
	inputs = Input((h,w,3))
	conv1 = Conv2D(16, (3,3), padding='same')(inputs)
	cm1 = context_module(conv1, 16)
	add1 = Add()([conv1, cm1]) # concat later
	conv2_stride = Conv2D(32, (3,3), strides=2, padding='same')(add1)
	cm2 = context_module(conv2_stride, 32)
	add2 = Add()([conv2_stride, cm2]) # concat later
	conv3_stride = Conv2D(64, (3,3), strides=2, padding='same')(add2)
	cm3 = context_module(conv3_stride, 64)
	add3 = Add()([conv3_stride, cm3]) # concat later
	conv4_stride = Conv2D(128, (3,3), strides=2, padding='same')(add3)
	cm4 = context_module(conv4_stride, 128)
	add4 = Add()([conv4_stride, cm4]) # concat later
	conv5_stride = Conv2D(256, (3,3), strides=2, padding='same')(add4)
	cm5 = context_module(conv5_stride, 256)
	add5 = Add()([conv5_stride, cm5])
	um1 = upsampling_module(add5, 128)
	concat1 = concatenate([um1, add4])
	lm1 = localization_module(concat1, 128)
	um2 = upsampling_module(lm1, 64)
	concat2 = concatenate([um2, add3])
	lm2 = localization_module(concat2, 64) # addup later
	um3 = upsampling_module(lm2, 32)
	concat3 = concatenate([um3, add2])
	lm3 = localization_module(concat3, 32) # addup later
	um4 = upsampling_module(lm3, 16)
	concat4 = concatenate([um4, add1])
	conv6 = Conv2D(32, (3,3), padding='same')(concat4) # addup later
	seg1 = Conv2D(1, (1,1), padding='same')(lm2)
	seg1 = UpSampling2D(size=(2,2))(seg1)
	seg2 = Conv2D(1, (1,1), padding='same')(lm3)
	sum1 = Add()([seg1, seg2])
	sum1 = UpSampling2D(size=(2,2))(sum1)
	seg3 = Conv2D(1, (1,1), padding='same')(conv6)
	sum2 = Add()([sum1, seg3])
	outputs = Activation('sigmoid')(sum2)
	network = tf.keras.Model(inputs = [inputs], outputs = [outputs])
	return network
```

## Switch to another file

All your files and folders are presented as a tree in the file explorer. You can switch from one to another by clicking a file in the tree.

## Rename a file

You can rename the current file by clicking the file name in the navigation bar or by clicking the **Rename** button in the file explorer.

## Delete a file

You can delete the current file by clicking the **Remove** button in the file explorer. The file will be moved into the **Trash** folder and automatically deleted after 7 days of inactivity.

## Export a file

You can export the current file by clicking **Export to disk** in the menu. You can choose to export the file as plain Markdown, as HTML using a Handlebars template or as a PDF.


# Synchronization

Synchronization is one of the biggest features of StackEdit. It enables you to synchronize any file in your workspace with other files stored in your **Google Drive**, your **Dropbox** and your **GitHub** accounts. This allows you to keep writing on other devices, collaborate with people you share the file with, integrate easily into your workflow... The synchronization mechanism takes place every minute in the background, downloading, merging, and uploading file modifications.

There are two types of synchronization and they can complement each other:

- The workspace synchronization will sync all your files, folders and settings automatically. This will allow you to fetch your workspace on any other device.
	> To start syncing your workspace, just sign in with Google in the menu.

- The file synchronization will keep one file of the workspace synced with one or multiple files in **Google Drive**, **Dropbox** or **GitHub**.
	> Before starting to sync files, you must link an account in the **Synchronize** sub-menu.

