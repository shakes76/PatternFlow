# Contents
* ## Description
     This part bulids the training model and finally saves the model's training parameters (h5 file). The size of the input images is (150,150), and batch size is 32. The use of callbacks is to stop the training process when the loss function has not changed significantly. Convolutional neural network is the algorithm of this model. It contains 1 input layer, 4 convolutional layers, 4 max pooling layers, dropout layer and fully connected layer. The output is 2 classes. It is used to classify left and right knees.
     
   Model structure and parameters:
   
   ![](https://github.com/1665446266/PatternFlow/blob/topic-recognition/recognition/project(%20OAI%20AKOA%20knee%20dataset)/bulid%20model/model%20structure.png?raw=true)

* ## Results 
   Training and validation accuracy:
   
   ![](https://github.com/1665446266/PatternFlow/blob/topic-recognition/recognition/project(%20OAI%20AKOA%20knee%20dataset)/bulid%20model/Training%20and%20validation%20accuracy.png?raw=true)
   
   Training and validation loss:
   
   ![](https://github.com/1665446266/PatternFlow/blob/topic-recognition/recognition/project(%20OAI%20AKOA%20knee%20dataset)/bulid%20model/Training%20and%20validation%20loss.png?raw=true)
   
  
* ## Code
   
   ```python

        __author__ = 'xiaomeng cai'
        __date__ = '10/15/2020 '

        import tensorflow as tf
        from tensorflow import keras

        #bulid model
        model= keras.models.Sequential()
        model.add(keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(512, activation='relu'))
        model.add(keras.layers.Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',optimizer=keras.optimizers.RMSprop(lr=1e-4),metrics=['acc'])

        # data generator
        train_dir = r'C:\Users\s4549082\Downloads\dataset\train'
        validation_dir = r'C:\Users\s4549082\Downloads\dataset\validation'

        train_datagen =keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        validation_datagen =keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=(150, 150),
                batch_size=32,
                class_mode='binary')

        validation_generator = validation_datagen.flow_from_directory(
                validation_dir,
                target_size=(150, 150),
                batch_size=32,
                class_mode='binary')

         #fit model
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
         
        # early stoping
        es = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=20)
        history = model.fit(
                  train_generator,
                  steps_per_epoch=12000/32,
                  epochs=1500,
                  validation_data=validation_generator,
                  validation_steps=150,
                  callbacks=[es]
        )

        # save model
        model.save('trained_model.h5')

        # plot Training and validation accuracy& Training and validation loss curve
        import matplotlib.pyplot as plt
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()
  ```



