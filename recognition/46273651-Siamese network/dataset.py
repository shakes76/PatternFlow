import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf

class data_loader():
    def __init__(self, path, label):
        self.path = path
        self.label = label

        self.data = np.array([])
        self.labels = np.array([])
        self.load_data()
        self.load_labels()

    def load_data(self):
        directory = os.listdir(self.path)
        self.data = np.array([plt.imread(self.path + '/' + i) for i in directory])
        self.data = np.expand_dims(self.data, axis=3)

    def load_labels(self):
        if self.label == 'AD':
            self.labels = np.ones(self.data.shape[0])
        elif self.label == 'NC':
            self.labels = np.zeros(self.data.shape[0])

    def get_data(self):
        return self.data

    def get_labels(self):
        return self.labels

class data_preprocess():
    def __init__(self, AD, NC):
        self.AD = AD
        self.NC = NC

        self.data = np.concatenate((self.AD.get_data(), self.NC.get_data()), axis=0)
        self.labels = np.concatenate((self.AD.get_labels(), self.NC.get_labels()), axis=0)
        self.tf_ds = tf.data.Dataset.from_tensor_slices((self.data, self.labels))

        self.tf_ds = self.resize(self.tf_ds)
        self.tf_ds = self.normalize(self.tf_ds)
        self.tf_ds = self.shuffle(self.tf_ds)
        self.tf_ds = self.batch(self.tf_ds, 8)

    def resize(self, data):
        return data.map(lambda x, y: (tf.image.resize(x, (224, 224)), y))

    def normalize(self, data):
        return data.map(lambda x, y: (x / 255.0, y))

    def shuffle(self, data):
        return data.shuffle(len(self.data))
    
    def batch(self, data, batch_size):
        return data.batch(batch_size)

    def get_data(self):
        return self.tf_ds

    def get_triplet_data(self):
        anchor = tf.data.Dataset.from_tensor_slices(self.AD.get_data())
        positive = tf.data.Dataset.from_tensor_slices(self.AD.get_data())
        negative = tf.data.Dataset.from_tensor_slices(self.NC.get_data())

        anchor = anchor.map(lambda x: tf.image.resize(x, (224, 224)))
        positive = positive.map(lambda x: tf.image.resize(x, (224, 224)))
        negative = negative.map(lambda x: tf.image.resize(x, (224, 224)))

        anchor = anchor.map(lambda x: x / 255.0)
        positive = positive.map(lambda x: x / 255.0)
        negative = negative.map(lambda x: x / 255.0)

        anchor = anchor.shuffle(len(self.AD.get_data()))
        positive = positive.shuffle(len(self.AD.get_data()))
        negative = negative.shuffle(len(self.NC.get_data()))

        self.shuffle(anchor)
        self.shuffle(positive)
        self.shuffle(negative)

        dataset = tf.data.Dataset.zip((anchor, positive, negative))

        # split the data into train and validation 
        train_dataset = dataset.take(round(self.AD.get_data().shape[0] * 0.8))
        val_dataset = dataset.skip(round(self.AD.get_data().shape[0] * 0.8))

        # batch the data
        train_dataset = train_dataset.batch(8)
        train_dataset = train_dataset.prefetch(8)

        val_dataset = val_dataset.batch(8)
        val_dataset = val_dataset.prefetch(8)

        return train_dataset, val_dataset

    def split_data(self, data):
        train_size = int(0.8 * len(data))

        train_data = data.take(train_size)
        val_data = data.skip(train_size)

        self.batch(train_data, 8)
        self.batch(val_data, 8)

        return train_data, val_data

if __name__ == '__main__':
    AD = data_loader('AD_NC/train/AD', 'AD')
    NC = data_loader('AD_NC/train/NC', 'NC')
    data = data_preprocess(AD, NC)
    train_data, val_data = data.split_data(data.get_data())

    print(train_data)
    print(val_data)

    training_data, validation_data = data.get_triplet_data()
    print(training_data)
    print(validation_data)