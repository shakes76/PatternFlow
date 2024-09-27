import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
tf.random.Generator = None
import tensorflow_addons as tfa
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.models import load_model

# Build context module in encoder part
def context_module(input, filter):

    out = InstanceNormalization()(input)
    out = LeakyReLU(alpha=0.01)(out)
    out = Conv2D(filter,(3,3),padding='same')(out)
    out = Dropout(0.3)(out)
    out = InstanceNormalization()(out)
    out = LeakyReLU(alpha=0.01)(out)
    out = Conv2D(filter,(3,3),padding='same')(out)
    
    return out

# Build upsampling module in decoder part
def upsampling_module(input, filter):
    out = UpSampling2D()(input)
    out = Conv2D(filter,(3,3),padding='same')(out)
    
    return out

# Build localization module in decoder part
def localization_module(input, filter):
    out = Conv2D(filter,(3,3),padding='same')(input)
    out = Conv2D(filter,(1,1),padding='same')(out)
    
    return out


def bulid_model():
    inputs = Input(shape = (height, width, channels))
    
    # Encoder
    c1 = Conv2D(16, (3, 3),padding='same') (inputs)
    cm1 = context_module(c1, 16)
    a1 = add([c1,cm1])
    c2 = Conv2D(32, (3, 3), strides = 2, padding='same')(a1)
    cm2 = context_module(c2, 32)
    a2 = add([c2,cm2])
    c3 = Conv2D(64, (3, 3), strides = 2, padding='same')(a2)
    cm3 = context_module(c3, 64)
    a3 = add([c3,cm3])
    c4 = Conv2D(128, (3, 3), strides = 2, padding='same')(a3)
    cm4 = context_module(c4, 128)
    a4 = add([c4,cm4])
    c5 = Conv2D(256, (3, 3), strides = 2, padding='same')(a4)
    cm5 = context_module(c5, 256)
    a5 = add([c5,cm5])   

    # Bridge
    u1 = upsampling_module(a5, 128)
    con1 = concatenate([u1, a4])

    # Decoder
    l1 = localization_module(con1, 128)
    u2 = upsampling_module(l1, 64)
    con2 = concatenate([u2, a3])
    l2 = localization_module(con2, 64)
    u3 = upsampling_module(l2, 32)
    con3 = concatenate([u3, a2])

    l3 = localization_module(con3, 32)
    u4 = upsampling_module(l3, 16)
    con4 = concatenate([u4, a1])

    c6 = Conv2D(32, (3, 3),padding='same') (con4)

    # Add segmentation part
    s1 = Conv2D(1, (1,1), padding='same')(l2)
    up1 = UpSampling2D()(s1)
    s2 = Conv2D(1, (1,1), padding='same')(l3)
    a6 = add([up1,s2])
    up2 = UpSampling2D()(a6)
    s3 = Conv2D(1, (1,1), padding='same')(c6)
    a7 = add([up2,s3])

    # Outputs
    outputs = Conv2D(1, (1, 1), activation = 'sigmoid') (a7)

    return Model(inputs=[inputs], outputs=[outputs])



if __name__ == "__main__":
    model = build_model()
    model.summary()
 
