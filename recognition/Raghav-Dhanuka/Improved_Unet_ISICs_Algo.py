#!/usr/bin/env python
# coding: utf-8
"""
##################################################
## Author: Raghav Dhanuka - s4593673
## Copyright: Copyright 2020, Improved U-Net - ISICs Dataset
## Credits: [Raghav Dhanuka, Shakes and Team]
## Date: Oct 31 12:17:14 2020
## License: COMP3701
## Version: “0.1.0”
## Mmaintainer: Raghav Dhanuka
## Email: r.dhanuka@uqconnect.edu.au
## Status: 'Dev'
## Description:  All the Modules for the Improved U-Net are present below which will be run using the driver script main.py
##################################################
"""


def path_for_dataset(path_train,path_seg):
    """
    This Function is used Extracting the file names of the images and masks in training, test and validation folders
    """
    
    isic_train = next(os.walk(path_train))[2] # returns all the files "DIR."
    isic_seg_train=next(os.walk(path_seg))[2] # returns all the files "DIR."

    print("No. of images in training folder= ",len(isic_train))
    print("No. of images in test folder= ",len(isic_seg_train))
    return isic_train, isic_seg_train





def sorted_test(isic_train, isic_seg_train):
    """
    This Function is used for Sorting the data with respect to labels
    """
    isic_train_sort=sorted(isic_train) # Sorting of data with respect to labels
    isic_seg_train_sort=sorted(isic_seg_train) # Sorting of data with respect to labels
    
    return isic_train_sort, isic_seg_train_sort





def Load_img(inp_path,isic):
    """ 
    This function is used for Loading the images from the Training_Input_x2 folder
    """
    # " - Storing them with the above dimensions specified"
    # " - Loading the images in Greayscale format"
    # " - Normalizing the image with 255 as Normalising data by dividing it by 255 should improve activation functions performance"
    # " - Sigmoid function works more efficiently with data range 0.0-1.0."
    
    X_ISIC_train= np.zeros((len(isic),img_height,img_width,1),dtype=np.float32)
    for n, id_ in tqdm_notebook(enumerate(isic), total=len(isic)): # capture all the images ids using tqdm
        img = load_img(inp_path+id_, color_mode = 'grayscale')  # Load images here
        x_img = img_to_array(img) # Convert images to array
        x_img = resize(x_img,(256,256,1),mode = 'constant',preserve_range = True)
        X_ISIC_train[n] = x_img/255 # Normalize the images
        
    return X_ISIC_train




def Load_segmentation(inp_path,isic):
    """ 
    This function is used for Loading the images from the Training_GroundTruth_x2 folder
    """
    # " - Storing them with the above dimensions specified"
    # " - Loading the images in Greayscale format"
    
    Y_ISIC_train= np.zeros((len(isic),img_height,img_width,1),dtype=np.uint8)
    for n, id_ in tqdm_notebook(enumerate(isic), total=len(isic)):
        # Load images
        img = load_img(inp_path+id_,color_mode = 'grayscale')
        x_img = img_to_array(img)
        x_img = resize(x_img,(256,256,1),mode = 'constant', preserve_range = True)
        Y_ISIC_train[n] = x_img
        
    return Y_ISIC_train




def load_dataset(X_ISIC_train, Y_ISIC_train):
    """
    This Function is used for spliting the dataset into Train, Test, and Val
    """
    X_train, X_test, y_train, y_test = train_test_split(X_ISIC_train, Y_ISIC_train, test_size=0.20, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    X_train.shape
    
    return X_train, X_test, y_train, y_test, X_val, y_val





def pre_processing(y_train,y_test,y_val):
    """
    This Function is used for Pre_processing the Y labels into two categoraical format
    """
    # " - By calculating the quotient Normalizing the image with data range 0.0-1.0"
    # " - By using One-Hot Encoding to Labels"
    Y_ISIC_train_sc = y_train//255
    Y_ISIC_test_sc = y_test//255
    Y_ISIC_val_sc = y_val//255
    Y_ISIC_train_cat = to_categorical(Y_ISIC_train_sc) # one hot encoding on Y_train
    Y_ISIC_test_cat = to_categorical(Y_ISIC_test_sc) # one hot encoding on Y_test
    Y_ISIC_val_cat = to_categorical(Y_ISIC_val_sc) # one hot encoding on Y_val
    
    return Y_ISIC_train_cat, Y_ISIC_test_cat, Y_ISIC_val_cat





# Dice Coeffient
from keras import backend as K
def dice_coeff(y_true, y_pred, smooth=1):
    """
    This Function is used to gauge similarity of two samples
    """
    # " - When applied to Boolean data, using the definition of true positive (TP), false positive (FP), and false negative (FN)"
    
    intersect = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersect
    coeff_dice = K.mean((intersect + smooth) / (union + smooth), axis=0)
    return coeff_dice




def dice_coeff_for_each_seg(y_true, y_pred, smooth=1):
    """
    This Function is used to gauge similarity of two samples
    """
    # " - When applied to Boolean data, using the definition of true positive (TP), false positive (FP), and false negative (FN)"
    # " - This will calculate the dice score foe each segmented data"
    
    intersect = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersect
    coeff_dice_ind = (intersect + smooth) / (union + smooth)
    return dice_coeff_for_each_seg




def generat_unet():
    """
    This Function is using the improved Unet Architecture with the following changes
    """
    # " - change activation to LeakyReLU from batchnorm"
    # " - change dropout layer from Dropout(0.05) to Dropout(0.3)"
    # " - Adding two Context module with a dropout layer in between two conv2d layer"
    # " - Performing the elementwise summation"
    # " - Downsampling the layer between two context module"
    # " - concatenating layer with corresponding downsampling layer"
    # " - Elemetwise summation of segmentation layers"
    
    # assigning input for the Unet model
    inputs = Input(shape=(256, 256, 1))
    CL = Conv2D(16, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(inputs)
    #context module - Preactivation of residual block with a dropout layer
    CL1 = Conv2D(16, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(CL)
    p1 = Dropout(0.3)(CL1)
    CL1 = Conv2D(16, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(p1)
    #Performing element wise summation
    CL1 = Add()([CL, CL1])

    #downsampling layer between two context modules
    CL1_ds = Conv2D(32, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same', strides=(2, 2))(CL1)

    CL2 = Conv2D(32, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(CL1_ds)
    p2 = Dropout(0.3)(CL2)
    CL2 = Conv2D(32, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(p2)
    
    CL2 = Add()([CL1_ds, CL2])    
    CL2_ds = Conv2D(64, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same', strides=(2, 2))(CL2)
    
    CL3 = Conv2D(64, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(CL2_ds)
    p3 = Dropout(0.3)(CL3)
    CL3 = Conv2D(64, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(p3)
    
    CL3 = Add()([CL2_ds, CL3])    
    CL3_ds = Conv2D(128, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same', strides=(2, 2))(CL3)
    
    CL4 = Conv2D(128, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(CL3_ds)
    p4 = Dropout(0.3)(CL4)
    CL4 = Conv2D(128, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(p4)
    
    CL4 = Add()([CL3_ds, CL4])    
    CL4_ds = Conv2D(256, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same', strides=(2, 2))(CL4)
    
    CL5 = Conv2D(256, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(CL4_ds)
    p5 = Dropout(0.3)(CL5)
    CL5 = Conv2D(256, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(p5)
    
    CL5 = Add()([CL4_ds, CL5])

    L6 = UpSampling2D()(CL5)

    #concatenating layer with corresponding downsampling layer
    C1 = concatenate([L6, CL4])

    UP1 = Conv2D(128, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(C1)
    UP1 = Conv2D(128, (1, 1), activation=LeakyReLU(alpha=0.01), padding='same')(UP1)
    L7 = UpSampling2D()(UP1)
    C2 = concatenate([L7, CL3])
    UP2 = Conv2D(64, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(C2)
    UP2 = Conv2D(64, (1, 1), activation=LeakyReLU(alpha=0.01), padding='same')(UP2)
    S1 = Conv2D(4, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(UP2)
    S1 = UpSampling2D()(S1)
    L8 = UpSampling2D()(UP2)
    C3 = concatenate([L8, CL2])
    UP3 = Conv2D(32, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(C3)
    UP3 = Conv2D(32, (1, 1), activation=LeakyReLU(alpha=0.01), padding='same')(UP3)
    S2 = Conv2D(4, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(UP3)
    L9 = UpSampling2D()(UP3)
    C4 = concatenate([L9, CL1])
    L10 = Conv2D(32, (1, 1), activation=LeakyReLU(alpha=0.01), padding='same')(C4)

    S3 = Conv2D(4, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(L10)

    #element wise summation of segmented layers
    addseg12 = Add()([S1, S2])
    addseg12 = UpSampling2D()(addseg12)
    addseg123 = Add()([addseg12, S3])
    outputs = Conv2D(2, (1, 1), activation="sigmoid")(addseg123)
    model = Model(inputs=inputs, outputs=outputs)
    
    return model





def loss_plot(results):
    """
    This Function is used for used for plotting the graph of the training and Validation loss with respect to epoch
    """
    plt.figure(figsize=(8, 8))
    plt.title("Binary_Crossentropy_loss")
    plt.plot(results.history["loss"], label="training_loss")
    plt.plot(results.history["val_loss"], label="validation_loss")
    plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("loss")
    plt.legend();





def acc_plot(results):
    """
    This Function is used for Plotting the training and validation accuracy with respect to epochs
    """
    plt.figure(figsize=(8,8))
    plt.title("Classification Accuracy")
    plt.plot(results.history["accuracy"],label="training_accuracy")
    plt.plot(results.history["val_accuracy"],label="validation_accuracy")
    plt.plot(np.argmin(results.history["val_accuracy"]),np.max(results.history["val_accuracy"]),marker="x",color="r",label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend();





def best_model(model,X_test,y_test,Y_ISIC_test_cat):
    """
    This Function is used for Capturing the Best model for the epoch
    """
    model.load_weights('model-ISIC.h5')
    test_preds=model.predict(X_test,verbose=1) # predict the model
    test_preds_max=np.argmax(test_preds,axis=-1) # Returns the indices of the maximum values along an axis
    print("Overall dice coefficient of the ISIC test data\n")
    print(dice_coeff(Y_ISIC_test_cat,test_preds))
    print(dice_coeff_for_each_seg(Y_ISIC_test_cat[1:5],test_preds[1:5]))
    n,h,w,g=y_test.shape
    test_preds_reshape=test_preds_max.reshape(n,h,w,g)
    return test_preds_reshape





def plot_ISIc(X, y, Y_pred,ix=None):
    """
    This function is used for ploting the True image vs the Predictive image from the above model
    """
    if ix is None:
        ix = random.randint(0, len(X))
    else:
        ix = ix
    
    fig, ax = plt.subplots(1, 3, figsize=(20, 10))
    ax[0].imshow(X[ix, ..., 0], cmap='gray')
    ax[0].contour(X[ix].squeeze(), colors='k', levels=[0.5])
    ax[0].set_title('Input Image')
    
    
    ax[1].imshow(y[ix, ..., 0], cmap='gray')
    ax[1].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[1].set_title('True Image')
    
    ax[2].imshow(Y_pred[ix, ..., 0], cmap='gray')
    ax[2].contour(Y_pred[ix].squeeze(), colors='k', levels=[0.5])
    ax[2].set_title('Predicted Image')
    






