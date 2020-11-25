from model import *
from dice import *
from tensorflow.keras import datasets, layers, models


# do prediction on test data - x 
# use function adv_model to create a model named unetmodel
# input: x_test
# output: predict_y

def model_prediction(x_test,model_name):
    predict_y = model_name.predict(x_test, verbose=0)
    return predict_y



    
