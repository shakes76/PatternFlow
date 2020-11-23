from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import os


def create_model():
    covn_base = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    covn_base.trainable = True
    # Freeze the front layer and train the last seven layers
    for layers in covn_base.layers[:-7]:
        layers.trainable = False
    # Building models
    model = tf.keras.Sequential()
    model.add(covn_base)
    model.add(tf.keras.layers.GlobalAveragePooling2D())  # Adding global average pooling layer
    model.add(tf.keras.layers.Dense(512, activation='relu'))  # Add full connectivity layer
    model.add(tf.keras.layers.Dropout(rate=0.5))  # Add dropout layer to prevent over fitting
    model.add(tf.keras.layers.Dense(2, activation='softmax'))  # Add output layer(2 categories)
    model.summary()  # Print parameter information of each layer

    # Compiling model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  # Using Adam optimizer, the learning rate is 0.001
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),  # Cross entropy loss function
                  metrics=["accuracy"])  # evaluation function
    return model

image_path = "/home/lbd855/DeepLearning/Knee/"
train_dir = image_path + "train"
validation_dir = image_path + "test"

train_image_generator = ImageDataGenerator(rescale=1. / 255,
                                           rotation_range=40,
                                           width_shift_range=0.2,
                                           height_shift_range=0.2,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True,
                                           fill_mode='nearest')

train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
                                                           batch_size=batch_size,
                                                           shuffle=True,
                                                           target_size=(im_height, im_width),
                                                           class_mode='categorical')

total_train = train_data_gen.n
validation_image_generator = ImageDataGenerator(rescale=1. / 255)  # normalization

val_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir,
                                                              batch_size=batch_size,
                                                              shuffle=False,
                                                              target_size=(im_height, im_width),
                                                              class_mode='categorical')
total_val = val_data_gen.n
# use tf.keras.applications And use the official pretraining model

model = create_model()
# Callback function 1: learning rate attenuation
reduce_lr = ReduceLROnPlateau(
    monior = 'val_loss', # Value to monior
    factor = 0.1 # The learning rate decreased to 1 /10 of the original
    patience = 2,
    # When the number of epoch passes and the performance of the model is notimproved,
    # the action of reducing the learning rate will be triggered
    mode = 'auto',
    # When the moitouing value is val_ When ACC,  the mode should be max, when the monitering value is val_
    # In case of loss, the mode should be min, in auto mode, the evaluation criteria are automatically inferred
    # from the name of the monitored value
    verbose = 1 # If true, a message is output for each update. Default value:false
)

# Callback function 2: save optimal model
checkpoint = ModelCheckpoint(
    filepath = './save_weights/myweights.h5', # Path to save the model
    monitor = 'val_acc', # Values to monitor
    save_weights_only = True,
    # If set to true, only the model weight is saved; otherwise ,the whole model(including model structure,
    # configuration information,etc.)will be saved
    save_best_only = True, # When set ro true, the current model is saved only when the monitoring value is improved
    mode = 'auto',
    # When the monitoring value is val_acc, the mode should be max; when the monitoring value is val_loss,
    # the mode should be min, in the auto mode, the evaluation criteria are automatically inferred from the name
    # of the monitored value
    period = 1 # The number of epoch in the interval between checkpoints
)
# Start training
history = model.fit(x = train_data_gen, # Input training set
                    steps_per_epoch = total_train // batch_size # The number of training steps contained in an epoch
                    epochs = epochs, # Training mode iterations
                    validation_data = val_data_gen # Input validation set
                    validation_steps = total_val // batch_size, # The number of training steps contained in an epoch
                    callbacks = [checkpoint,reduce_lr]) # Execute callback function
# save the trained model weight
model.save_weights('./save_weights/myweights.h5')

# Record the accuracy and loss value of training set and verification set
history_dist = history.history
train_loss = history_dict["loss"] # loss value of training set
train_accuracy = history_dict["accuracy"] # accuracy of the training set
val_loss = history_dict["val_loss"] # loss value of validation set
val_accuracy = history_dict["val_accuracy"] # accuracy of validation set

    
plt.figure()
plt.plot(range(epochs),train_loss,label = 'train_loss')
plt.plot(range(epochs),val_loss, label = 'val_loss')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')
