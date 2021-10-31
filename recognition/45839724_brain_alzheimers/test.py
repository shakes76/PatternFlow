import enum
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import math
import time
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.python.keras.engine import input_layer
import tensorflow_addons as tfa
tf.compat.v1.enable_eager_execution()

def save_data():
    # LOAD IN DATA. Organize by patient, to prevent data leakage. 

    DIR = r"C:\Users\hmunn\OneDrive\Desktop\COMP3710\Project\Data\AKOA_Analysis\\"
    file_paths = [DIR + x for x in os.listdir(DIR)]
    new_patient_ids = {} # key: e.g. OAI9014797_3_L, value: new id
    data = {} # key: unique patient id (created), value: ([xdata], [labels [0 for left, 1 for right]])
    totals = 0
    right = 0
    for file in file_paths:
        is_right = "RIGHT" in file or "Right" in file or "right" in file or "R_I_G_H_T" in file
        right += 1 if is_right else 0
        totals += 1
        info = file.split("_BaseLine_")
        patient_id = info[0] + "_" + info[1].split("de3")[0].split("_")[0] + "_" + ("L" if not is_right else "R")
        if patient_id not in new_patient_ids:
            new_patient_ids[patient_id] = len(new_patient_ids)
        new_id = new_patient_ids[patient_id]
        img = np.asarray(Image.open(file).convert("L"))
        img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
        label = 1 if is_right else 0
        if new_id in data:
            data[new_id][0].append(img)
            data[new_id][1].append(label)
        else:
            data[new_id] = ([img], [label])

    # SPLIT DATA. Get train/test split based on patients. 

    TEST_SPLIT = 0.4
    num_patients = len(data)
    patient_ids = list(range(0, num_patients))
    test_patients = random.sample(patient_ids, int(num_patients*TEST_SPLIT))
    train_patients = [x for x in patient_ids if x not in test_patients]

    xtrain, xtest, ytrain, ytest = [], [], [], []
    for pid in patient_ids:
        #print(data[pid])
        for idx in range(len(data[pid][0])):
            if pid in train_patients:
                xtrain.append(data[pid][0][idx])
                ytrain.append(data[pid][1][idx])
            else:
                xtest.append(data[pid][0][idx])
                ytest.append(data[pid][1][idx])
    print(len(xtrain), len(xtest), len(ytrain), len(ytest))
    del data


    # SHUFFLE DATA AND SAVE. 

    indices_train = list(range(0, len(xtrain)))
    indices_test = list(range(0, len(xtest)))
    random.shuffle(indices_train)
    random.shuffle(indices_test)
    xtrain = np.array(xtrain)
    xtrain = xtrain[indices_train]
    np.save("xtrain", xtrain)
    del xtrain
    xtest = np.array(xtest)
    xtest = xtest[indices_test]
    np.save("xtest", xtest)
    del xtest
    ytrain = np.array(ytrain)
    ytrain = ytrain[indices_train]
    np.save("ytrain", ytrain)
    del ytrain
    ytest = np.array(ytest)
    ytest = ytest[indices_test]
    np.save("ytest", ytest)
    del ytest


SAVE_DATA = False

if SAVE_DATA:
    save_data()
else:
    xtrain = np.load(r"xtrain.npy")
    xtrain = xtrain[0:2000]
    xtrain = np.reshape(xtrain, (*xtrain.shape, 1))
    ytrain = np.load(r"ytrain.npy")
    ytrain = ytrain[0:2000]

    xval = np.load(r"xtrain.npy")
    xval = xval[250:700]
    xval = np.reshape(xval, (*xval.shape, 1))
    yval = np.load(r"ytrain.npy")
    yval = yval[250:700]

# GET FOURIER FEATURES FOR POSITIONAL ENCODINGS. 

# img_data: tensor of shape (datapoints, rows, cols)
def get_positional_encodings(img_data, bands=4, sampling_rate=10):
    ''' (Doesn't work, slightly off.)
    # assume 2 dimensions, using single channel images
    data_points, rows, cols = img_data.shape
    xr, xc = tf.linspace(-1,1,rows), tf.linspace(-1,1,cols)
    xd = tf.expand_dims(tf.reverse(tf.meshgrid(xr,xc), axis=[-3]),3)
    xd = tf.reshape(tf.concat([xd[0], xd[1]], axis=2),(rows,cols,2))
    xd = tf.repeat(tf.expand_dims(xd, -1), repeats=[2*bands + 1], axis=3) # (rows, cols, 2, 2F + 1)
    # logscale for frequencies ( * pi) , 0 start as 10**0 = 1
    frequencies = tf.experimental.numpy.logspace(0.0,(tf.math.log(sampling_rate/2)/tf.math.log(10.)), num = bands, dtype = tf.float32) * math.pi
    # (228,260,2,9)
    f_features = tf.cast(xd, tf.float32)
    f_features = tf.concat([tf.math.sin(f_features[:,:,:,0:4] * frequencies), tf.math.cos(f_features[:,:,:,4:8] * frequencies), tf.expand_dims(f_features[:,:,:,8], -1)], axis=-1)
    f_features = tf.repeat(tf.reshape(f_features, (1,rows,cols,2*(2*bands + 1))), repeats=[data_points],axis=0) # (data_points, 228, 260, 18)
    f_features = tf.cast(f_features, tf.float32)
    return tf.reshape(tf.concat((tf.expand_dims(tf.cast(img_data, tf.float32), 3),f_features),axis=-1), (data_points, rows*cols, -1)) # add data in and flatten images
    '''
    # Fourier Encoding partially from https://github.com/Rishit-dagli/Perceiver 
    b, *axis, _ = img_data.shape
    axis_pos = list(map(lambda size: tf.linspace(-1.0, 1.0, num=size), axis))
    pos = tf.stack(tf.meshgrid(*axis_pos, indexing="ij"), axis=-1)
    x = tf.expand_dims(pos, -1)
    x = tf.cast(x, dtype=tf.float32)
    orig_x = x
    scales = tf.experimental.numpy.logspace(
        0.0,
        math.log(sampling_rate / 2) / math.log(10.),
        num=bands,
        dtype=tf.float32
    )
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]
    x = x * scales * math.pi
    x = tf.concat([tf.math.sin(x), tf.math.cos(x)], axis=-1)
    x = tf.concat((x, orig_x), axis=-1)
    encoding = tf.repeat(tf.reshape(x, (1, 228, 260, 2 * (2 * bands + 1))), repeats = b, axis=0)
    return tf.reshape(tf.concat((img_data, encoding), axis=-1), (b, 228*260, -1))  

# DEFINE MODELS

def get_attention_module(channel_size, data_size, latent_size):

    inputss = [layers.Input((latent_size, channel_size)),layers.Input((data_size, channel_size))]

    # Q, K & V linear networks
    query_mlp = inputss[0]
    query_mlp = layers.LayerNormalization()(query_mlp)
    latent_output = query_mlp
    query_mlp = layers.Dense(channel_size)(query_mlp)

    key_mlp = inputss[1]
    key_mlp = layers.LayerNormalization()(key_mlp)
    key_mlp = layers.Dense(channel_size)(key_mlp)

    value_mlp = inputss[1]
    value_mlp = layers.LayerNormalization()(value_mlp)
    value_mlp = layers.Dense(channel_size)(value_mlp)

    # QKV cross-attention
    attention_module = layers.Attention(use_scale=True)([query_mlp, key_mlp, value_mlp])
    attention_module = layers.Dense(channel_size)(attention_module)
    attention_module = layers.Add()([latent_output, attention_module])
    attention_module = layers.LayerNormalization()(attention_module)

    # New query from attention module 
    new_latent = layers.Dense(channel_size, activation=tf.nn.gelu)(attention_module)
    #new_latent = layers.Dense(channel_size, activation=tf.nn.gelu)(new_latent)
    new_latent = layers.Dense(channel_size)(new_latent)
    new_latent = layers.Add()([attention_module, new_latent])

    cross_attention = keras.Model(inputs=inputss, outputs = new_latent)
    return cross_attention

def get_transformer_module(latent_size, channel_size, transformer_heads):
    latent_input = layers.Input((latent_size, channel_size))
    layer_init = latent_input
    for i in range(4): # 4 transformer blocks
        transformer = layers.LayerNormalization()(layer_init)
        transformer = layers.MultiHeadAttention(num_heads = transformer_heads, key_dim = channel_size)(transformer, transformer, \
            return_attention_scores = False)
        transformer = layers.Add()([latent_input, transformer])
        transformer = layers.LayerNormalization()(transformer)
        
        new_query = layers.Dense(channel_size, activation=tf.nn.gelu)(transformer)
        #new_query = layers.Dense(channel_size, activation=tf.nn.gelu)(new_query)
        new_query = layers.Dense(channel_size)(new_query)
        transformer = layers.Add()([new_query, transformer])
        layer_init = transformer

    return keras.Model(inputs = latent_input, outputs = transformer)

def truncated_initializer(shape, dtype=None):
    norm = tf.random.normal(shape, mean=0.0, stddev=0.02, dtype=dtype)
    # truncation
    return tf.math.minimum(tf.math.maximum(norm, tf.constant(-2, dtype=tf.float32, shape=norm.shape)),tf.constant(2, dtype=tf.float32, shape=norm.shape)) 

def create_batches_from_data(xdata, ydata, batches):
    xdata_new = xdata[:int(len(xdata) / batches) * batches]
    ydata_new = ydata[:int(len(ydata) / batches) * batches]
    return (xdata_new, ydata_new)

class Perceiver(tf.keras.Model):
    def __init__(self, latent_size = 64, data_size = 228*260, bands = 4, transformer_heads = 4, 
                sampling_rate = 10, iterations = 4):
        super(Perceiver, self).__init__()
        self.bands = bands
        self.latent_size = latent_size
        self.data_size = data_size
        self.transformer_heads = transformer_heads
        self.channel_size = 2*(2*bands + 1) + 1 # data (1) + 2 dim * (2F + 1)
        self.sampling_rate = sampling_rate
        self.iterations = iterations
        self.init_latent = self.add_weight(shape=(self.latent_size, self.channel_size), initializer= truncated_initializer, trainable=True)
        self.init_latent = tf.reshape(self.init_latent, (1,*self.init_latent.shape))
        self.attention_module = get_attention_module(channel_size=self.channel_size, data_size=self.data_size, latent_size=self.latent_size)
        self.transformer_module = get_transformer_module(self.latent_size, self.channel_size, self.transformer_heads)
        self.global_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.classify = layers.Dense(1, activation='sigmoid')# binary crossentropy

    def call(self, xdata):
        encoded_data = get_positional_encodings(xdata, bands=self.bands, sampling_rate=self.sampling_rate)
        input_data = [self.init_latent, encoded_data] # fix input
        for layer_num in range(self.iterations):
            new_latent = self.attention_module(input_data)
            new_query = self.transformer_module(new_latent)
            input_data[0] = new_query
        return self.classify(self.global_pool(new_query))
        
    def train_model(self, xtrain, ytrain, xval, yval, epochs, batches, lr_func): 
        X_train, y_train = create_batches_from_data(xtrain, ytrain, batches)
        X_val, y_val = create_batches_from_data(xval, yval, batches)

        history = self.fit(
			X_train, y_train,
			epochs = epochs,
			batch_size = batches,
            callbacks=[lr_func],
            validation_data = (X_val, y_val),
            validation_batch_size = batches
		)

        # Plotting the Learning curves
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.legend(['training', 'validation'], loc='upper left')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss (Binary Cross-Entropy)')
        plt.xlabel('Epochs')
        plt.legend(['training', 'validation'], loc='upper left')
        plt.show()

def learning_rate_decay(epoch):
    lr = 0.001
    decay_epochs = [84, 102, 114]
    for ep in decay_epochs:
        if epoch >= ep:
            lr /= 10
    return (lr)

perceiver = Perceiver()
learning_rate_fnc = tf.keras.callbacks.LearningRateScheduler(learning_rate_decay)

# Compile the model
perceiver.compile(
	optimizer = tfa.optimizers.LAMB(learning_rate=0.001, weight_decay_rate = 0.0002),
	loss = tf.keras.losses.BinaryCrossentropy(),
	metrics = tf.keras.metrics.BinaryAccuracy(name="accuracy")
)
 
perceiver.train_model(xtrain, ytrain, xval, yval, epochs=6, batches=8, lr_func=learning_rate_fnc)

# perceiver.save("./perceiver_model")

del xtrain
del xval
del ytrain
del yval

# evaluate model

xtest, ytest = np.load(r"xtest.npy")[0:1000], np.load(r"ytest.npy")[0:1000]
xtest = np.reshape(xtest, (*xtest.shape, 1))
xtest, ytest = create_batches_from_data(xtest, ytest, 8)


'''
# Uncomment to predict 8 random images.
import random
choices = [random.choice(list(range(len(xtest)))) for i in range(8)]
imgs = np.array([xtest[j] for j in choices])
imgs = imgs[:int(len(imgs) / 8) * 8]
print(imgs.shape)
predictions = perceiver.predict(imgs)

for idx, p in enumerate(predictions):
	plt.imshow(imgs[idx])
	print("prediction:", p, "actual:", ytest[choices[idx]])
	plt.show()
'''

# Uncomment for test accuracy (avoids filling up vram)
_, acc = perceiver.evaluate(xtest, ytest, batch_size=8)
print(f"Accuracy on test set:{round(acc * 100, 4)}%")
