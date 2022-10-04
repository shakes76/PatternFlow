## Components for the network

import tensorflow as tf
import keras

# Classes
class path:
    def __init__(self, train, val, test):
        self.train = train
        self.val = val
        self.test = test

img_path = path(
    '/Users/wotah_man/Documents/UQ/ISIC Dataset/Train',
    '/Users/wotah_man/Documents/UQ/ISIC Dataset/Val',
    '/Users/wotah_man/Documents/UQ/ISIC Dataset/Test'
)

# Functions
def normalize(dataset):
    '''normalize the image data to 0~1 float'''
    for img, lbl in dataset:
        x = tf.math.divide(img, 255.0)
        y = lbl
    return x, y

def backbone():
    '''the backbone model'''
    pass

def RPN(featuremap):
    '''the RPN model'''

    initializer = tf.keras.initializers.GlorotNormal(seed=None)
    input_= tf.keras.layers.Input(shape=[None, None, featuremap.shape[-1]], name='rpn_INPUT')

    shared = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu', strides=1, name='rpn_conv_shared',kernel_initializer=initializer)(input_)
    # 5*2:  5 different size * 1 scale anchor, 2 label probabilities
    x = tf.keras.layers.Conv2D(5*1*2 , (1, 1), padding='valid', activation='linear',name='rpn_class_raw',kernel_initializer=initializer)(shared) 

    rpn_class_logits = tf.keras.layers.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)
    rpn_probs = tf.keras.layers.Activation("softmax", name="rpn_class_xxx")(rpn_class_logits) # --> BG/FG

    # Bounding box refinement. [batch, H, W, depth] 5*4:  5 different size * 1 scale anchor, 4 delta coordinates
    x = tf.keras.layers.Conv2D(5*1*4, (1, 1), padding="valid", activation='linear', name='rpn_bbox_pred',kernel_initializer=initializer)(shared) 

    # Reshape to [batch, anchors, 4]
    rpn_bbox = tf.keras.layers.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)
    outputs = [rpn_class_logits, rpn_probs, rpn_bbox]
    rpn = tf.keras.models.Model(input_, outputs, name="RPN")

    return rpn

def rpn_loss(rpn_logits,rpn_deltas, gt_labels,gt_deltas , indices, batchlen):
    '''
    rpn_logits,rpn_deltas: the predicted logits/deltas to all the anchors
    gt_labels,gt_deltas: the correct labels and deltas to the chosen training anchors
    indices: the indices of the chosen training anchors
    '''
    
    predicted_classes = tf.gather_nd(rpn_logits, indices)
    foregroundindices=indices[gt_labels.astype('bool')] #labels: 0:BG  1:FG
    
    predicted_deltas=tf.cast(tf.gather_nd(rpn_deltas, foregroundindices),tf.float32) #only the foreground anchors contribute to the box loss
    gt_deltas=tf.cast(tf.gather_nd(gt_deltas, foregroundindices),tf.float32)


    lf=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    classloss = lf(gt_labels,predicted_classes)
    classloss=tf.reduce_mean(classloss)
    
    deltaloss=tf.losses.huber_loss(gt_deltas,predicted_deltas)
    deltaloss=tf.reduce_mean(deltaloss)
    
    return classloss,deltaloss

def YOLO(input_layer, NUM_CLASS):
    '''The YOLO Network Container'''

    #params
    depth = 64
    kernal = (7, 7)

    conv = keras.layers.Conv2D(depth, kernal, activation='LeakyReLU', padding='same')(input_layer)

def compile():
    '''Compile the model'''

    pass

def lossfunc():
    '''loss funtion for the YOLO network'''
    pass
