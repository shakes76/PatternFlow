from methods import *
from model import *
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(gpus[0], True)

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))



import matplotlib.pyplot as plt

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels=None, batch_size=1, dim=(256,256,128), n_channels=1,
                 n_classes=None, shuffle=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(tf.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        seed = 42
        tf.random.set_seed(seed)
        'Updates indexes after each epoch'
        self.indexes = tf.range(len(self.list_IDs))
        if self.shuffle == True:
            self.indexes = tf.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        # X = np.empty((self.batch_size, *self.dim, self.n_channels))
        X = []
        Y = []

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            
            # Store mri case
            x = process_nii('./dataset/semantic_MRs_anon/' + ID + '_LFOV.nii.gz', norma=True)
            X.append(x)


            # Store mri label
            y = process_nii('./dataset/semantic_labels_anon/' + ID + '_SEMANTIC_LFOV.nii.gz')
            Y.append(y)

        X = tf.stack(X)
        Y = tf.stack(y)
        return X, Y


if __name__ == "__main__":
    target_dir = "./dataset/semantic_MRs_anon"
    train_list, val_list, test_list = train_val_test_split(target_dir)
        

    # Generators
    train_generator = DataGenerator(train_list[6:])
    validation_generator = DataGenerator(val_list)
    test_generator = DataGenerator(test_list)
    print(len(train_generator), len(validation_generator), len(test_generator))

    model = Unet_3d()

    learning_rate = 0.001
    epochs = 1
    decay_rate = 0.0000001
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate, decay = decay_rate), loss="sparse_categorical_crossentropy", metrics=[dice_coef])
    print(model.summary())
    # model.compile(
    # optimizer='sgd',
    # loss=dice_coefficient,
    # metrics=dice_coefficient,
    # )

    history = model.fit(train_generator, epochs=15, validation_data=validation_generator,shuffle=False)
    model.save("./results/models/unet1")

    _ = plt.plot(history.history['loss'])
    _ = plt.plot(history.history['val_loss'])
    _ = plt.title('Train & Validation loss')
    _ = plt.ylabel('loss')
    _ = plt.xlabel('epoch')
    _ = plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig('./results/images/unet1_loss.png')

    test_results = model.evaluate(test_generator, verbose=0)
    print(test_results)



