from tensorflow.kears.layers import Input, Conv2D, Add, LeakyReLU, ZeroPadding2D, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import random_normal

def compose(*funcs):
    return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)

'''
    build a darknet convlution block which has conv layer, normalize later abd activation layer
'''
def DarkNetConv_Block(*args, **kwargs):

    # build a darknet Conv2D layer
    conv_kwargs = {'kernel_initializer': random_normal(stddev=0.02), 'kernal_regularizer':l2(5e-4), 'use_bias': False}
    conv_kwargs['padding'] = 'valid' if kwargs.get('stride')==(2,2) else 'same'
    conv_kwargs.update(kwargs)
    return compose(
        Conv2D(*args, **conv_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1)
    )

'''
    build reshape block body for darknet 53
'''
def reshape_body(tensor, num_filters, num_blocks):
    tensor = ZeroPadding2D(((1,0),(1,0)))(tensor)
    tensor = DarkNetConv_Block(num_filters, (3,3), strides=(2,2))(tensor)

    for i in range(num_blocks):
        conv_out = DarkNetConv_Block(num_filters//2, (1,1))(tensor)
        conv_out = DarkNetConv_Block(num_filters, (3,3))(conv_out)
        tensor = Add()([tensor, conv_out])
    return tensor

def DarkNet(tensor):
    # use 32 filters for first darknet conv block
    tensor = DarkNetConv_Block(32, (3,3))(tensor)

    # five reshape steps with third, forth and fifth reshape blocks have one output each
    # first reshape block body
    tensor = reshape_body(tensor, 64, 1)

    # second reshape block body
    tensor = reshape_body(tensor, 128, 2)

    # third reshape block body, get first feature map
    feature1 = reshape_body(tensor, 256, 8)

    # forth reshape block body, get second feature map
    feature2 = reshape_body(feature1, 512, 8)

    # fifth reshape block body, get third feature map 
    feature3 = reshape_body(feature2, 1024, 4)

    return feature1, feature2, feature3

