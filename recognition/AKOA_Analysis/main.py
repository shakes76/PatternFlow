import numpy as np
from sklearn.model_selection import train_test_split
import glob
import PIL
from PIL import Image
from lateral_classification import train

def main():
    
    imgdir = 'imgs/*.png'
    epochs = 10

    #Import the data

    filelist = glob.glob(imgdir)
    image = np.array([np.array(Image.open(fname)) for fname in filelist])
    
    #Create label

    label = []

    #Populate label list

    for fname in filelist:
        if 'right' in fname.lower():
            label.append(0)
        elif 'R_I_G_H_T' in fname:
            label.append(0)
        elif 'left' in fname.lower():
            label.append(1)
        elif 'L_E_F_T' in fname:
            label.append(1)
            
    #Convert label list to numpy array

    label = np.array(label)

    #Split label and images into test and training sets

    X_train, X_test, y_train, y_test = train_test_split(image, label, test_size=0.25, random_state=42)
    
    #Get shape of first file
    shape = image[0].shape
    
    train(shape, epochs, X_train, y_train, X_test, y_test)

if __name__ == '__main__':
    main()



