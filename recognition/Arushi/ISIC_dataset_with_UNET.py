"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ===========================================================================================================
Author: Arushi Mahajan
Student Number: 45755833
Copyright: Copyright 2020, UNet with ISIC dataset
Credits: Arushi Mahajan, Shakes and Team
License: COMP3710
Version: 1.0.1
Maintainer: Arushi Mahjan
Email: arushi.mahajan@uqconnect.edu.au
Status: Dev
Date Created: 31/10/2020
Date Modified: 05/11/2020
Description: This file contains all the functions that are required to run the test driver script
# ===========================================================================================================
"""


def load_dataset(train_input_path, train_groundtruth_path):  
    """This function will load the dataset"""
    isic_features = next(os.walk(train_input_path))[2] 
    isic_labels = next(os.walk(train_groundtruth_path))[2] 
    return isic_features, isic_labels


def sorting_labels(isic_features, isic_labels):  
    """This function will sort the data with respect to labels"""
    isic_features_sort = sorted(isic_features) 
    isic_labels_sort = sorted(isic_labels)     
    return isic_features_sort, isic_labels_sort


# get and resize the training input dataset
def load_features(inp_path, ids):
    """ This function loads the data from training input folder into grayscale mode 
    and normalizes the features to 0 and 1 """
    X_isic_train = np.zeros((len(ids), img_height, img_width, 1), dtype = np.float32)
    for n, id_ in tqdm_notebook(enumerate(ids), total = len(ids)): # capture all the images ids using tqdm       
        # load the image
        img = load_img(inp_path + id_, color_mode = 'grayscale') 
        x_img = img_to_array(img) # convert images to array
        x_img = resize(x_img, (256,256,1), mode = 'constant', preserve_range = True)
        X_isic_train[n] = x_img/255 # normalize the images
    return X_isic_train 


# function for loading the training groundtruth dataset
def load_labels(inp_path, ids):
    """ This function loads the data from training groundtruth folder into grayscale mode """
    y_isic_train = np.zeros((len(ids), img_height, img_width,1), dtype = np.uint8)
    for n, id_ in tqdm_notebook(enumerate(ids), total = len(ids)): # capture all the images ids using tqdm 
        # load the image
        img = load_img(inp_path + id_,color_mode = 'grayscale') 
        x_img = img_to_array(img) # convert images to array
        x_img = resize(x_img,(256,256,1),mode = 'constant', preserve_range = True)
        y_isic_train[n] = x_img
    return y_isic_train



# train-validation-test split
def split_datatset(X_isic_train, y_isic_train):
    """This function is performing train, validation and test split and splitting the data 60:20:20 ratio"""
    X_train, X_test, y_train, y_test = train_test_split(X_isic_train, y_isic_train, test_size = 0.20, random_state = 42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.25, random_state = 42)    
    return X_train, X_test, y_train, y_test, X_val, y_val



def encoding(y_train,y_test,y_val):
    """This function is normalizing the segmentation data and performing one hot encoding"""
    # normalizing the labels to 0 and 1 
    y_train_sc = y_train//255
    y_test_sc = y_test//255
    y_val_sc = y_val//255
    # one hot encoding 
    y_train_encode = to_categorical(y_train_sc) 
    y_test_encode = to_categorical(y_test_sc) 
    y_val_encode = to_categorical(y_val_sc) 
    return y_train_encode, y_test_encode, y_val_encode



def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    """ This function is used to add 2 convolutional layers with the parameters passed to it"""
    
    # first layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size), kernel_initializer = "he_normal", padding = "same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x) # it draws samples from a truncated normal distribution centered on 0 to normalize the outputs from previous layers
    x = Activation("relu")(x)
    
    # second layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size), kernel_initializer = "he_normal", padding = "same")(input_tensor)
    if batchnorm:
            x = BatchNormalization()(x) # it draws samples from a truncated normal distribution centered on 0 to normalize the outputs from previous layers
    x = Activation("relu")(x)    
    return x




def get_unet(input_img, n_filters = 16, dropout = 0.1, batchnorm = True):
    """ This function is used to generate and define the U-Net architecture - Encoder and Decoder"""
    
    # contracting path 
    c1 = conv2d_block(input_img, n_filters = n_filters*1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2)) (c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters = n_filters*2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2)) (c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters = n_filters*4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2)) (c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters = n_filters*8, kernel_size = 3, batchnorm = batchnorm)
    p4 = MaxPooling2D((2, 2)) (c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters = n_filters*16, kernel_size = 3, batchnorm = batchnorm)
    
    # expansive path 
    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides = (2, 2), padding = 'same') (c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters = n_filters*8, kernel_size = 3, batchnorm = batchnorm)

    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides = (2, 2), padding = 'same') (c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters = n_filters*4, kernel_size = 3, batchnorm = batchnorm)

    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides = (2, 2), padding = 'same') (c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters = n_filters*2, kernel_size = 3, batchnorm = batchnorm)

    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides = (2, 2), padding = 'same') (c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters = n_filters*1, kernel_size = 3, batchnorm = batchnorm)
    
    outputs = Conv2D(2, (1, 1), activation = 'sigmoid') (c9)
    model = Model(inputs = [input_img], outputs = [outputs])
    return model



def dice_coeffient(y_true, y_pred, smooth = 1):
    """ This function is used to find the dice coefficient of overall model """
    intersect = K.sum(K.abs(y_true * y_pred), axis = [1,2,3])
    union = K.sum(y_true,[1,2,3]) + K.sum(y_pred,[1,2,3]) - intersect
    coeff_dice = K.mean((intersect + smooth) / (union + smooth), axis = 0)
    return coeff_dice



def dice_loss(y_true, y_pred, smooth = 1):
    """ This function is used to evaluate dice loss """
    return 1 - dice_coeffient(y_true, y_pred, smooth = 1)


def dice_coeffient_each(y_true, y_pred, smooth=1): 
    """ This function is used to find the dice coefficient for each test segmentation """
    intersect = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersect
    coeff_dice_ind = (intersect + smooth) / (union + smooth)
    return dice_coeffient_each



def lossPlot(results):
    """Function is used to plot for training loss and validation loss with respect to epochs"""
    plt.figure(figsize = (8, 8))
    plt.title("dice loss")
    plt.plot(results.history["loss"], label = "training_loss")
    plt.plot(results.history["val_loss"], label = "validation_loss")
    plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker = "x", color = "r", label = "best model")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend();




def accuracyPlot(results):
    """Function is used to plot for training accuracy and validation accuracy with respect to epochs"""
    plt.figure(figsize = (8,8))
    plt.title("Classification Accuracy")
    plt.plot(results.history["accuracy"],label = "training_accuracy")
    plt.plot(results.history["val_accuracy"],label = "validation_accuracy")
    plt.plot(np.argmin(results.history["val_accuracy"]),np.max(results.history["val_accuracy"]),marker = "x",color = "r",label = "best model")
    plt.xlabel("Epochs")
    plt.legend();





def best_model(model,X_test,y_test,y_test_encode):
    """Function for loading the best model and predicting on model"""
    model.load_weights('ISIC_model.h5')
    test_preds = model.predict(X_test,verbose=1) 
    test_preds_max = np.argmax(test_preds,axis=-1)
    print("Overall dice coefficient of the ISIC test data\n")
    print(dice_coeffient(y_test_encode,test_preds)))
    n,h,w,g = y_test.shape
    test_preds_reshape = test_preds_max.reshape(n,h,w,g)
    return test_preds_reshape




def plot_ISIC(X, y, Y_pred,ix=None):
    """ This function returns the input image, true image and predicted image"""
    if ix is None:
        ix = random.randint(0, len(X))
    else:
        ix = ix   

    fig, ax = plt.subplots(1, 3, figsize=(20, 10))
    
    #Plotting original image
    ax[0].imshow(X[ix, ..., 0], cmap='gray')
    ax[0].contour(X[ix].squeeze(), colors='k', levels=[0.5])
    ax[0].set_title('Input Image')   
    
    #Plotting true image
    ax[1].imshow(y[ix, ..., 0], cmap='gray')
    ax[1].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[1].set_title('True Image')
    
    #Plotting predicted image
    ax[2].imshow(Y_pred[ix, ..., 0], cmap='gray')
    ax[2].contour(Y_pred[ix].squeeze(), colors='k', levels=[0.5])
    ax[2].set_title('Predicted Image')    
