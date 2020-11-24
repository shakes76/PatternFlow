import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K 
import PIL

# convert training images to array
def convert_array(filelist):
    data = []
    for fname in filelist:
        image = np.asarray(PIL.Image.open(fname))
        image = tf.image.resize(image, (256,256))
        data.append(image)
    data = np.array(data, dtype=np.float32)
    return data

# convert ground truth images to array
def convert_array_truth(filelist):
    data = []
    for fname in filelist:
        image = np.asarray(PIL.Image.open(fname))
        image = image[:,:,np.newaxis]
        image = tf.image.resize(image, (256,256), method = 'nearest')
        data.append(image)
    data = np.array(data, dtype=np.uint8)
    return data

# compile and fit model
def fit(model,x,y, epoch_size, batch):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
                     loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                     metrics=['accuracy'])

    model.fit(x, y, epochs=epoch_size, batch_size=batch,
                    validation_split=0.2)

# calculate dice coefficient
def dice_coef(y_true, y_pred, smooth=1.):
    y_true = tf.convert_to_tensor(y_true, dtype='float32')
    y_pred = tf.convert_to_tensor(y_pred, dtype='float32')
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# display results
def display(display_list):
    plt.figure(figsize=(15, 15))
    title= ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()
