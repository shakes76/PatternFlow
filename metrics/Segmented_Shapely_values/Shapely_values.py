import tensorflow as tf
import tensorflow.keras as keras
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import skimage.segmentation as seg
import itertools
from math import factorial

def Shapely_values(segnumb = 15,segchoice =None,model=None,data=None,groundtruth = None):
    if not (Model and Data and groundtruth):
        model,data,groundtruth = defult_model()
    if not segchoice:
        image_slic = seg.slic(data[0],n_segments=segnumb)
    patches = tf.range(max(image_slic))


    def shapely_step(ele):
        elements = patches[tf.range(mex(patches))!=ele]
        result = 0
        i = 1
        while i < max(patches):
            it = intertool.combinations(elements,i)
            for j in it:
                factor = (factorial(len(j)) + factorial(nCk(patches,i) - len(j) - 1))/factorial(nCk(patches,i))
                resultwithoutp = model.predict(data[data in j.extend(i)])
                result = model.predict(data[data in j])
                result += factor*(resultwithoutp-result)
        
        return result

    f = tf.function(shapely_step)

    ret = tf.zeros_like(patches)
    ret = tf.map_fn(f, patches, axis=1)
    print(ret)

#         def image_show(image, nrows=1, ncols=1, cmap='gray'):
#             fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
#             ax.imshow(image, cmap='gray')
#             ax.axis('off')
#             return fig, ax

# #need to change the alpha and create a colour list as well
#     image_show(color.label2rgb(image_slic, image, kind='overlay'))

def nCk(n,k): 
    return int(reduce(lambda x,y:x*y, ((n-i)/(i+1) for i in range(k)), 1)) 
    

def defult_model():

    (x_train, y_train), (x_eval, y_eval) = cifar10.load_data()
    train_labels = keras.utils.to_categorical(y_train)
    eval_labels = keras.utils.to_categorical(y_eval)



    x_train = x_train / 255.0
    x_eval = x_eval / 255.0

    x_test = x_eval[:1000]
    x_eval = x_eval[1000:]
    y_test = eval_labels[:1000] 
    eval_labels = eval_labels[1000:] 

    print(x_train.shape[1:])
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
                    activation='relu',input_shape = x_train.shape[1:]))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2),strides =(2,2)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1),
                    activation='relu',input_shape = x_train.shape[1:]))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2),strides =(2,2)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Conv2D(256, kernel_size=(5, 5), strides=(1, 1),
                    activation='relu',input_shape = x_train.shape[1:]))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2),strides =(2,2)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dropout(0.7))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dropout(0.7))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dropout(0.7))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(10, activation='softmax',name="out"))
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adam(lr=0.001),
                metrics=['accuracy'])
    #model.predict(x_train)
    class AccuracyHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.acc = []

        def on_epoch_end(self, batch, logs={}):
            self.acc.append(logs.get('val_acc'))
    history = AccuracyHistory()
    model.fit(x_train,train_labels, batch_size=256, 
            epochs=150,
            verbose=1,
            validation_data=(x_eval, eval_labels),
            callbacks=[history])
    score = model.evaluate(x_test, y_test, verbose=1)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model_json = model.to_json()
    with open("cifar10model.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights("cifar10model.h5")
    print("Saved model to disk")
    return model,x_test[0],ytest[0]