from tensorflow.keras import layers, models, optimizers, preprocessing


class KneeModel:
    # model initialization
    def __init__(self, train_path, val_path, test_path):
        # train data set path
        self.train_path = train_path
        # validation data set path
        self.val_path = val_path
        # test data set path
        self.test_path = test_path
        # image resize
        self.image_height = 224
        self.image_width = 224
        self.batch_size = 32
        self.model = self.constructModel()

    def constructModel(self):
        # 3 dimension input
        input_image = layers.Input(shape=(self.image_height, self.image_width, 3))

        # total 4 conv_pool unit
        # convolution
        conv1 = layers.Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding="same")(input_image)
        # bn layer, accelerate train
        bn1 = layers.BatchNormalization(axis=3)(conv1)
        # relu layer
        relu1 = layers.LeakyReLU(alpha=0.3)(bn1)
        # max pool
        pool1 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(relu1)

        conv2 = layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding="same")(pool1)
        bn2 = layers.BatchNormalization(axis=3)(conv2)
        relu2 = layers.LeakyReLU(alpha=0.3)(bn2)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(relu2)

        conv3 = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same")(pool2)
        bn3 = layers.BatchNormalization(axis=3)(conv3)
        relu3 = layers.LeakyReLU(alpha=0.3)(bn3)
        pool3 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(relu3)

        conv4 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same")(pool3)
        bn4 = layers.BatchNormalization(axis=3)(conv4)
        relu4 = layers.LeakyReLU(alpha=0.3)(bn4)
        pool4 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(relu4)

        # batch_size * height * width * 3 to batch_size * length
        flatten = layers.Flatten()(pool4)
        # full connection
        full = layers.Dropout(0.5)(layers.Dense(64, activation='relu')(flatten))
        # softmax output
        output = layers.Dense(2, activation='softmax')(full)

        model_ = models.Model(input_image, output)
        # model compile
        model_.compile(optimizer=optimizers.Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
        model_.summary()
        return model_

    def fit(self):
        # image data preprocessing and generator
        data_generator = preprocessing.image.ImageDataGenerator(rescale=1. / 255)
        # generate image from path, label is the subdir
        train_generator = data_generator.flow_from_directory(
            self.train_path,
            target_size=(self.image_height, self.image_width),
            batch_size=self.batch_size,
            seed=0)

        # same with train generator
        val_generator = data_generator.flow_from_directory(
            self.val_path,
            target_size=(self.image_height, self.image_width),
            batch_size=self.batch_size,
            seed=0)

        # history save record of model training, train images are from train_generator, every epoch train 200 steps,
        # once a epoch has trained, then evaluate on val_generator
        self.history = self.model.fit_generator(train_generator, steps_per_epoch=200, epochs=10,
                                                validation_data=val_generator, validation_steps=50)
        # save the model
        self.model.save('model.h5')

    def evaluate(self):
        data_generator = preprocessing.image.ImageDataGenerator(rescale=1. / 255)
        test_generator = data_generator.flow_from_directory(
            self.test_path,
            target_size=(self.image_height, self.image_width),
            batch_size=self.batch_size,
            seed=0)
        # score is a list, list[0] is loss, list[1] is accuracy
        score = self.model.evaluate_generator(test_generator)
        print("The accuracy of knee model on test data set: {:.2f}".format(score[1]))
        print("The loss of knee model on test data set: {:.2f}".format(score[0]))

    def load_model(self, model_path):
        self.model = models.load_model(model_path)
        self.history = self.model.history

