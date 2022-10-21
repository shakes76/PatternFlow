# buld model 
from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras import Model

"""build SRCNN model with three layers"""
def get_model():
    height, width = 128, 128
    input_img = Input((height, width, 3))
    lay = Conv2D(64, (9,9), padding = 'same', activation= 'relu')(input_img)
    lay = Conv2D(32, (1,1), padding = 'same', activation= 'relu')(lay)
    lay = Conv2D(3, (5,5), padding = 'same', activation= 'relu')(lay)
    model = Model(inputs=input_img, outputs=lay)
    model.summary()
    return model


