##Training the model

import tensorflow as tf
from keras import layers, models
import numpy as np
from modules import backbone
#import modules
import dataset

NUMOFTRAINDATAS=4000
NUMOFTESTDATAS=1000
BATCHLEN=5
BATCHPEREPOCH=NUMOFTRAINDATAS/BATCHLEN
PROPOSALCOUNT=20
ROISIZE=[5,5]
MASKROISIZE=[14,14]
CLASSDICT={0:'lesion'}

backboneNN = backbone()
backboneNN.summary()

backboneNN.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

backboneNN.fit(dataset.x_train, 
               dataset.y_train, epochs=20, batch_size=768,validation_data=(dataset.x_val, dataset.y_val))

#modules.train_rpn(rpnmodell=modules.RPN,fmmodel=modules.fmmodel,batchlen=5,epochs=20,numofdatas=NUMOFTRAINDATAS) 
