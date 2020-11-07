import tensorflow as tf
import matplotlib as plt
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from PIL import Image

import glob
import numpy as np
from dice import *
from data_prepare import *
from model import *
from predict import *



def main():
    # load image data and normalize them
    data_pre()
    # train the model
    model = adv_model(4) 
    history = model.fit(x_train_af, y_seg_train, epochs=10, batch_size=16,
                    validation_data=(x_validate_af, y_seg_validate))

    # do model prediction and calculate dice coefficient 
    

    # visual the outcome  (show the 10th image outcome)
    outcome_visual(x_test,x_seg_test,predict_y,10)

if __name__ == "__main__":
    main()