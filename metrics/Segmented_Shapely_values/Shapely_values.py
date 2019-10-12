import keras
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import skimage.segmentation as seg
import intertool

def Shapely_values(segnumb = 15,segchoice =None,model=None,data=None,groundtruth = None):
    if not (Model and Data and groundtruth):
        model,data,groundtruth = defult_model()
    if not segchoice:
        image_slic = seg.slic(data[0],n_segments=segnumb)
    patches = np.arange(np.max(image_slic))

    for i in patches
       it = intertool.comob(patches,i)
    for i in patches:
        result = 0
        for j in it:
            if not i in j:
                factor = (len(j)! + (len(it) - len(j) - 1)!)/len(it)!
                resultwithoutp = model.predict(data)
                result = model.predict(data)
                result += factor*(resultwithoutp-result)
        ret[i] = result

    
    

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

    # model_json = model.to_json()
    # with open("cifar10model.json", "w") as json_file:
    #     json_file.write(model_json)

    # model.save_weights("cifar10model.h5")
    # print("Saved model to disk")
    return model,x_test,y_test

# load json and create model
# from tensorflow.keras.models import model_from_json
# json_file = open('cifar10model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("cifar10model.h5")
# print("Loaded model from disk")
# loaded_model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adam(lr=0.001),
#               metrics=['accuracy'])
# score = model.evaluate(x_test, y_test, verbose=1)