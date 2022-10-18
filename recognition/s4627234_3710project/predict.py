
import os 
import SimpleITK as sitk
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import train
import dataset
import modules

def image_show(file_path, images_paths, image_num):
    ''' Show the image_num'th image from file_path with printing out its shape'''
    image_array=sitk.ReadImage(os.path.join(file_path,images_paths[image_num]))
    print(sitk.GetArrayFromImage(image_array).shape)
    plt.imshow(sitk.GetArrayFromImage(image_array))
    plt.show() 

def image_save():
    pass

val_x, val_y = dataset.load_val()
model = train.training(data_reshape = False)

pred = model.predict(val_x)
gt = tf.convert_to_tensor(val_y,dtype=tf.float32)
print (modules.DSC(gt,pred))

r = [1, 23, 45, 67, 78, 12, 5, 28]
for n in r:
    fig = plt.figure()

    fig.subplots_adjust(hspace = 0.4, wspace = 0.4)
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(val_x[r])
    ax.title.set_text("Testing Image")
    ax.axis('off')

    result=pred[r]>0.5
    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(result*255, cmap="gray")
    ax.title.set_text("Predict Image")
    ax.axis('off')

    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(val_y[r])
    ax.title.set_text("Ground Truth")
    ax.axis('off')

    plt.savefig('./images/output'+ str(n) +'.png')

