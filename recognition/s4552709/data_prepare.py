import tensorflow as tf
import glob
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt


filelist_train = glob.glob('C:/Users/s4552709/dataset/keras_png_slices_data/keras_png_slices_train/*.png')
# uses image data in the form of a NumPy array
x_train = np.array([np.array(Image.open(fname)) for fname in filelist_train])
print('x_train.shape',x_train.shape)

filelist_test = glob.glob('C:/Users/s4552709/dataset/keras_png_slices_data/keras_png_slices_test/*.png')
x_test = np.array([np.array(Image.open(fname)) for fname in filelist_test])
print('x_test.shape',x_test.shape)

filelist_validate = glob.glob('C:/Users/s4552709/dataset/keras_png_slices_data/keras_png_slices_validate/*.png')
x_validate = np.array([np.array(Image.open(fname)) for fname in filelist_validate])
print('x_validate.shape',x_validate.shape)


filelist_seg_train = glob.glob('C:/Users/s4552709/dataset/keras_png_slices_data/keras_png_slices_seg_train/*.png')
x_seg_train = np.array([np.array(Image.open(fname)) for fname in filelist_seg_train])
print('x_seg_train.shape',x_seg_train.shape)

filelist_seg_test = glob.glob('C:/Users/s4552709/dataset/keras_png_slices_data/keras_png_slices_seg_test/*.png')
x_seg_test = np.array([np.array(Image.open(fname)) for fname in filelist_seg_test])
print('x_seg_test.shape',x_seg_test.shape)

filelist_seg_validate = glob.glob('C:/Users/s4552709/dataset/keras_png_slices_data/keras_png_slices_seg_validate/*.png')
x_seg_validate = np.array([np.array(Image.open(fname)) for fname in filelist_seg_validate])
print('x_seg_validate.shape',x_seg_validate.shape)

n_train_samples, h, w = x_train.shape
print('n_train_samples, h, w',n_train_samples, h, w)



# 4 classes
#0 - Background
#1 - CSF (cerebrospinal fluid) 脑脊液
#2 - Gray matter
#3 - White matter

#normaliz
#x_train = tf.cast(x_train, tf.float32) / 255.0
#x_test = tf.cast(x_test, tf.float32) / 255.0
#x_validate = tf.cast(x_validate, tf.float32) / 255.0
x_train_norm = x_train/255
x_test_norm = x_test/255
x_validate_norm = x_validate/255

x_train_af = x_train_norm[:,:,:,np.newaxis]
x_test_af = x_test_norm[:,:,:,np.newaxis]
x_validate_af = x_validate_norm[:,:,:,np.newaxis]
print(x_train_af.shape)


x_seg_train = x_seg_train/85
x_seg_validate = x_seg_validate/85
x_seg_test = x_seg_test/85
print(x_seg_train.shape)


#Converts a class vector (integers) to binary class matrix.
y_seg_train = tf.keras.utils.to_categorical(x_seg_train, num_classes=4)
y_seg_validate = tf.keras.utils.to_categorical(x_seg_validate, num_classes=4)
y_seg_test = tf.keras.utils.to_categorical(x_seg_test, num_classes=4)
print(y_seg_train.shape)


# combine img and their lable image together
train_ds = tf.data.Dataset.from_tensor_slices((x_train_af, y_seg_train))
val_ds = tf.data.Dataset.from_tensor_slices((x_test_af, y_seg_validate))
test_ds = tf.data.Dataset.from_tensor_slices((x_validate_af, y_seg_test))


# change the sequance of every data set
train_ds = train_ds.shuffle(len(x_train))
val_ds = val_ds.shuffle(len(x_validate))
test_ds = test_ds.shuffle(len(x_test))


def display_pre(x_test, predict_y,num):
    # display a model's prediction
    mask = np.argmax(predict_y[num],axis = -1)
    mask = np.expand_dims(mask, axis = -1)
    # transfer the mask to 
    #img = ImageOps.autocontrast(img = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask)))
    #data_visual([x_test[num],predict_y[i],y_seg_test[i]])
   
    

def data_visual(visual_list):
    plt.figure(figsize =(10,10))
    for i in range(len(visual_list), i+1):
        plt.subplot(1,len(visual_list), i+1)
        plt.imshow(visual_list[i],camp='gray')
        plt.axis('off')
    plt.show()


def outcome_visual(x_test,x_seg_test,predict_y,i):
    #i can refer which image set show

    fig = plt.figure()
    # combine three image together 
    # orignal image ,predict image,  ground true image
    ax1 = fig.add_subplot(311)
    ax1.imshow(x_test[i],camp='gray')
    plt.xlabel('orignal image')
    ax2 = fig.add_subplot(312)
    ax2.imshow(x_seg_test[i],camp='gray')
    plt.xlabel('predict image')
    ax3 = fig.add_subplot(313)
    ax3.imshow(predict_y[i],camp='gray')
    plt.xlabel('ground true image')
    
    plt.show()




def main():
    unet_model = adv_model(4)
    
    # train the model
    history = unet_model.fit(x_train_af, y_seg_train, epochs=10, batch_size=16,
                    validation_data=(x_validate_af, y_seg_validate))

    # do model prediction and calculate dice coefficient 
    predict_y = model_prediction(x_test, unet_model)

    #show dice coefficient
    dice = dice_coefficient(y_seg_test, predict_y, smooth=0.0001)
    print("dice_coefficient",dice)

    # visual the outcome  (show the 10th image outcome)
    outcome_visual(x_test,x_seg_test,predict_y,10)

if __name__ == "__main__":
    main()