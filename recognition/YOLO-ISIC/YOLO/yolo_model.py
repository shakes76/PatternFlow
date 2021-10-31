from tensorflow.kears.layers import Input, Conv2D, Add, LeakyReLU, \
    ZeroPadding2D, BatchNormalization, UpSampling2D, Concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import random_normal
from tensorflow.keras.models import Model
from functools import reduce, warps


#---------------------------------------------------------------#
#
#                    DARKNET 53 STRUCTURE  
#     this is the main feature extraction structure of YOLO                    
#
#---------------------------------------------------------------#
def compose(*funcs):
    return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)

@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_initializer' : random_normal(stddev=0.02), 'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

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

#---------------------------------------------------------------#
#
#                    FNP STRUCTURE  
#     this is the strengthen feature extraction structure of YOLO                    
#
#---------------------------------------------------------------#

def ConvFeatureInFNP(tensor, num_filters):
    tensor = DarkNetConv_Block(num_filters, (1,1))(tensor)
    tensor = DarkNetConv_Block(num_filters * 2, (3,3))(tensor)
    tensor = DarkNetConv_Block(num_filters, (1,1))(tensor)
    tensor = DarkNetConv_Block(num_filters * 2, (3,3))(tensor)
    tensor = DarkNetConv_Block(num_filters, (1,1))(tensor)
    return tensor

def GenerateYOLOHead(tensor, num_filtes, out_filters):
    tensor = DarkNetConv_Block(num_filtes * 2, (3,3))(tensor)
    ### DARKNET CONV 2D
    tensor = DarknetConv2D(out_filters,(1,1))(tensor)
    return tensor

def YOLO(input_size, mask, num_classes):
    inputs = Input(input_size)

    feature1, feature2, feature3 = DarkNet(inputs)

    tensor = ConvFeatureInFNP(feature3, 512)
    yolo_head_3 = GenerateYOLOHead(tensor, 512, len(mask[0])*(num_classes + 5))

    tensor = compose(DarkNetConv_Block(256, (1,1)), UpSampling2D(2)(tensor))

    tensor = Concatenate()([tensor, feature2])

    tensor = ConvFeatureInFNP(tensor, 256)
    yolo_head_2 = GenerateYOLOHead(tensor, 256, len(mask[1]) * (num_classes + 5))

    tensor = compose(DarkNetConv_Block(128, (1,1)), UpSampling2D(2)(tensor))

    tensor = Concatenate()([tensor, feature1])

    tensor = ConvFeatureInFNP(tensor, 128)
    yolo_head_1 = GenerateYOLOHead(tensor, 128, len(mask[2]) * (num_classes + 5))

    return Model(inputs, [yolo_head_3, yolo_head_2, yolo_head_1])


#-------------------------------------------------------#
#
#            Warap all together
#
#-------------------------------------------------------#

def get_yolo_model(model_body, input_size, num_classes, anchros, masks):
    y_true = [Input(shape = (input_shape[0] // {0:32, 1:16, 2:8}[l], input_shape[1] // {0:32, 1:16, 2:8}[l], \
                                len(anchors_mask[l]), num_classes + 5)) for l in range(len(anchors_mask))]
    model_loss  = Lambda(
        yolo_loss, 
        output_shape    = (1, ), 
        name            = 'yolo_loss', 
        arguments       = {'input_shape' : input_shape, 'anchors' : anchors, 'anchors_mask' : anchors_mask, 'num_classes' : num_classes}
    )([*model_body.output, *y_true])
    model       = Model([model_body.input, *y_true], model_loss)
    return model
