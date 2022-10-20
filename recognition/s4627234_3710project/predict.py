
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



# analyse history of training the model
def analyse_training_history(history):
    """Plots the acuraccy and validation accuracy of the model as it trains."""
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()
analyse_training_history(history)

# plot some predictions
def display_predictions(model, ds, n=1):
    """Makes n predictions using the model and the given dataset and displays
    these predictions."""
    for image, mask in ds.take(n):
        pred_mask = model.predict(image[tf.newaxis, ...])
        pred_mask = tf.math.round(pred_mask)
        display([tf.squeeze(image), tf.squeeze(mask), tf.squeeze(pred_mask)])

display_predictions(model, test_ds, n=3)

# compute dice similarity coefficients predictions
def compute_dice_coefficients(model, ds):
    """Computes the average dice similarity coefficient for all predictions
    made using the provided dataset."""
    DCEs = []
    for image, mask in ds:
        pred_mask = model.predict(image[tf.newaxis, ...])
        pred_mask = tf.math.round(pred_mask)
        DCE = dice_coefficient(mask, pred_mask)
        DCEs.append(DCE)
    print("Average Dice Coefficient = ", sum(DCEs)/len(DCEs))
    
compute_dice_coefficients(model, test_ds)