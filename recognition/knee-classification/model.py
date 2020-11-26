import tensorflow as tf


class KneeClassifier:

    def __init__(self, img_shape, no_epochs, train_dataset, validation_dataset, test_dataset, learning_rate):
        self.img_shape = img_shape
        self.no_epochs = no_epochs
        self.train_dataset = train_dataset,
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.base_model = None
        self.complete_model = None
        self.learning_rate = learning_rate

    # create basic model from InceptionV3
    def create_base_model(self):
        base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet',
                                                       input_shape=self.img_shape, pooling='max')
        base_model.trainable = False
        self.base_model = base_model

    # Get model summary details
    def get_model_summary(self, model_type):
        if model_type == 'base':
            print(self.base_model.summary())
        if model_type == 'complete':
            print(self.complete_model.summary())

    # create complete TF DL model with input pre processing, data augmentation, dropout.
    # Used adam as optimizer and used binary cross-entropy loss because there are only two classes.
    def create_complete_model(self):

        self.create_base_model()

        prediction_layer = tf.keras.layers.Dense(1)
        preprocess_input = tf.keras.applications.inception_v3.preprocess_input
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
        ])

        inputs = tf.keras.Input(shape=self.img_shape)
        x = data_augmentation(inputs)
        x = preprocess_input(x)
        x = self.base_model(x, training=False)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = prediction_layer(x)
        model = tf.keras.Model(inputs, outputs)
        self.complete_model = model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate),
                                            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                                            metrics=['accuracy'])

    # This function can be used to evaluate the DL model in the beginning and the after the training.
    def model_evaluation(self, eval_type):
        dataset = None
        if eval_type == "initial":
            dataset = self.validation_dataset
        elif eval_type == "final":
            dataset = self.test_dataset

        if dataset is not None and self.complete_model is not None:
            loss, accuracy = self.complete_model.evaluate(dataset)
            if eval_type == 'initial':
                print("initial loss: {:.2f}".format(loss))
                print("initial accuracy: {:.2f}".format(accuracy))
            elif eval_type == "final":
                print("Test loss: {:.2f}".format(loss))
                print("Test accuracy: {:.2f}".format(accuracy))

    # Train the knee classifier
    def train_knee_classifier(self):

        self.create_complete_model()
        self.model_evaluation(eval_type='initial')

        history = self.complete_model.fit(self.train_dataset, epochs=self.no_epochs,
                                          validation_data=self.validation_dataset)

        return history

