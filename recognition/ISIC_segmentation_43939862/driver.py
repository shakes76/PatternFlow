import glob
import tensorflow as tf

#Load data
image_files = glob.glob('../../../ISIC2018_Data/ISIC2018_Images/*.jpg')
label_files = glob.glob('../../../ISIC2018_Data/ISIC2018_Labels/*.png')

split_ratio = 0.8
split_size = int(len(image_files)*split_ratio)

train_ds = tf.data.Dataset.from_tensor_slices((image_files[:split_size], 
                                               label_files[:split_size]))
test_ds = tf.data.Dataset.from_tensor_slices((image_files[split_size:], 
                                               label_files[split_size:]))

#Pre-process data
train_ds = train_ds.shuffle(split_size)
test_ds = test_ds.shuffle(len(image_files)-split_size)

def map_fn(image, label):
    img = tf.io.read_file(image)
    img = tf.io.decode_jpeg(img, channels = 0)
    img = tf.cast(img, tf.float32) / 255.0
    
    lbl = tf.io.read_file(label)
    lbl = tf.io.decode_png(lbl, channels = 0)
    lbl = tf.one_hot(lbl, depth = 2, axis = 2)
    lbl = tf.cast(lbl, tf.float32)
    lbl = tf.squeeze(lbl)
    return img, lbl

train_ds = train_ds.map(map_fn)
test_ds = test_ds.map(map_fn)

#Build model


#Compile model


#Train model


#Evaluate model


#Predictions