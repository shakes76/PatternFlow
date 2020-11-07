def data_import()
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
