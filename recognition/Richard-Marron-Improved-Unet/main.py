"""
    Main driver script for the
    improved U-Net model.
    
    author: Richard Marron
    status: Development
"""
import numpy as np
import glob
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from unetModule import ImprovedUNet

def normalise_images(image_set, ground_truth=False):
    """
    Normalise images between 0 and 1
        Params:
            image_set : Numpy array of images
            ground_truth : If the set is the ground truth data
            
        Return : Normalised version of image_set
    """
    if not ground_truth:
        # Normalise without rounding
        return image_set / 255.0
    else:
        # Round the normalisation since we want just 0 or 1 values
        return np.around(image_set / 255.0)

def load_images(path: str, ground_truth: bool=False, truncate: bool=False):
    """
    Get all images from a given path
        Params:
            path : Path to the dataset
            ground_truth : Whether the folder is the ground truth data
                           (These have images of format PNG)
            truncate : Whether to use the full dataset or only 1/4
        
        Return : Numpy array of images which are also Numpy arrays
    """
    if ground_truth:
        # Images have PNG format
        path += f"*.png"
    else:
        # Images have JPG format
        path += f"*.jpg"
    
    print("Loading image paths...")
    img_paths = glob.glob(path)
    if truncate:
        # Only take the first 1/4 of the images
        img_paths = img_paths[:len(img_paths)//4]
    print("Successfully loaded paths!") 
    # Read in images from path and return numpy array
    if not ground_truth:
        # Read in RGB and resize to 512x384
        print("Converting training images into numpy arrays...")
        return np.array([cv2.cvtColor(cv2.resize(cv2.imread(path, cv2.IMREAD_COLOR),
                                                 dsize=(512, 384)), 
                                      cv2.COLOR_BGR2RGB) for path in img_paths], dtype=np.float32)
    else:
        # Read in grayscale and resize to 512x384
        print("Converting ground truth images into numpy arrays...")
        return np.expand_dims(np.array([cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE),
                                                             dsize=(512, 384)) for path in img_paths], 
                                                 dtype=np.float32), 
                                        axis=-1)

def get_splits(input_set, truth_set):
    """
    Generates train/test/validation splits from the given dataset
        Params:
            image_set : Set of input images we want to get the splits from.
            truth_set : Set of ground truth images which act as labels
        
        Require : Image sets must have shape (N, h, w, c)
        
        Return: 
            in_train : The training portion of the data, including ground truth
             in_test : The testing portion of the data, including ground truth
            in_valid : The validation portion of the data, including ground truth
    """
    print("Splitting the data...")
    # Get the training and test splits
    in_train, in_test, truth_train, truth_test = train_test_split(input_set, truth_set,
                                                                  test_size=0.4, random_state=123)
    print("Got training set!\nGenerating test and validation...")
    # From the test split, get the validation split
    in_test, in_valid, truth_test, truth_valid = train_test_split(in_test, truth_test,
                                                                  test_size=0.6, random_state=123)
    
    return (in_train, truth_train), (in_test, truth_test), (in_valid, truth_valid)

def fit_or_load(model, train, valid, path:str, fit:bool=True):
    """
    Choose whether to fit the model or, if the model is already trained,
    load the saved weight
        Params:
            model     : The model we want to fit or load
            train     : The training dataset
            valid     : The validation dataset
            path      : Path to weights file
            fit       : True if we want to fit the model. False if we want to load
        
        Return: History of the training process if fit = True
    """
    if fit:
        # Fit and then save weights of the model
        hist = model.fit(train[0], train[1], 
                         validation_data=valid, batch_size=10, epochs=100)
        model.save_weights(path)
        return hist
    else:
        # Load weights into model
        model.load_weights(path)
        
def plot_results(model, test, hist=None):
    """
    Plot some of the predicted values from the model
        Params:
            model : The model that is being used to predict
            test  : The test dataset
            hist  : The history of the training process
    """
    p_test = model.predict(test[0], batch_size=10)
    fig, axs = plt.subplots(nrows=3, ncols=6)
    
    for i in range(0, 3):
        for j in range(0, 6, 3):
            axs[i, j].imshow(p_test[(j+1)*(2*i+1) - 10])
            axs[i, j].set_title("Predicted Mask")
            axs[i, j].axis("off")
            axs[i, j+1].imshow(test[1][(j+1)*(2*i+1) - 10])
            axs[i, j+1].set_title("True Mask")  
            axs[i, j+1].axis("off")
            axs[i, j+2].imshow(test[0][(j+1)*(2*i+1) - 10])
            axs[i, j+2].set_title("Reality")  
            axs[i, j+2].axis("off")  
    
    if hist is not None:
        # Plot the training curves
        plt.figure()
        plt.plot(hist.history["dice_function"], label="Training")
        plt.plot(hist.history["val_dice_function"], label="Validation")
        plt.legend(loc="upper left")
        plt.title("Dice Similarity Coefficient During Training")
        plt.xlabel("Epoch")
        plt.ylabel("Similarity Score")

def main(debugging=False):
    """
    Main program entry
        Params:
            debugging : When True, the data is truncated to 1/4
                        of original size so program runs faster
    """
    # Load normalised images
    input_images = normalise_images(load_images(path="./Data/ISIC2018_Task1-2_Training_Input_x2/", 
                                                ground_truth=False, truncate=debugging), ground_truth=False)
    # Load normalized ground truth segmentation
    gt_images = normalise_images(load_images(path="./Data/ISIC2018_Task1_Training_GroundTruth_x2/",
                                             ground_truth=True, truncate=debugging), ground_truth=True)
        
    # Print out some useful information
    print(f"Total input images: {input_images.shape}")
    print(f"Total ground truth images: {gt_images.shape}")
    print(f"Input image shape: {input_images[0].shape}")
    print(f"Ground truth image shape: {gt_images[0].shape}")
    # Show example of image and it's segmentation
    plt.imshow(input_images[0])
    plt.figure()
    plt.imshow(gt_images[0])
    
    # Create an instance of an Improved U-Net
    unet = ImprovedUNet(input_shape=input_images[0].shape)
    # Get the model
    unet_model = unet.model()
    # Have a look at the model summary
    unet_model.summary()
    # Compile the model
    unet_model.compile(optimizer=unet.optimizer,
                       loss=unet.loss,
                       metrics=unet.metrics)
    
    # Generate train/test/validation sets
    train, test, valid = get_splits(input_images, gt_images)
    print(f"Train Set Shape \t: {train[0].shape}")
    print(f"Test Set Shape  \t: {test[0].shape}")
    print(f"Validation Set Shape \t: {valid[0].shape}")
    
    hist = fit_or_load(unet_model, train, 
                       valid, path="./weights/test.h5", fit=True)
    
    # Test model on the test set
    unet_model.evaluate(test[0], test[1], batch_size=10)
    
    plot_results(unet_model, test, hist)
    
    plt.show()

if __name__ == "__main__":
    main(debugging=False)