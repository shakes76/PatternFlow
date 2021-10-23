import tensorflow as tf
from tensorflow.keras import layers

def model_unet(filters):
    inputs = tf.keras.layers.Input(shape=(256, 256, 3))
      
    block1 = layers.Conv2D(filters, 3, padding='same', activation='relu')(inputs)
    block1 = layers.Conv2D(filters, 3, padding='same', activation='relu')(block1)
    
    block2 = layers.MaxPooling2D()(block1)
    block2 = layers.Conv2D(2*filters, 3, padding='same', activation='relu')(block2)
    block2 = layers.Conv2D(2*filters, 3, padding='same', activation='relu')(block2)
    
    block3 = layers.MaxPooling2D()(block2)
    block3 = layers.Conv2D(4*filters, 3, padding='same', activation='relu')(block3)
    block3 = layers.Conv2D(4*filters, 3, padding='same', activation='relu')(block3)
      
    block4 = layers.MaxPooling2D()(block3)
    block4 = layers.Conv2D(8*filters, 3, padding='same', activation='relu')(block4)
    block4 = layers.Conv2D(8*filters, 3, padding='same', activation='relu')(block4)
                               
    block5 = layers.MaxPooling2D()(block4)
    block5 = layers.Conv2D(16*filters, 3, padding='same', activation='relu')(block5)
    block5 = layers.Conv2D(16*filters, 3, padding='same', activation='relu')(block5)
    
    block6 = layers.UpSampling2D()(block5)
    block6 = layers.concatenate([block6, block4])
    block6 = layers.Conv2D(8*filters, 3, padding='same', activation='relu')(block6)
    block6 = layers.Conv2D(8*filters, 3, padding='same', activation='relu')(block6)
    
    block7 = layers.UpSampling2D()(block6)
    block7 = layers.concatenate([block7, block3])
    block7 = layers.Conv2D(4*filters, 3, padding='same', activation='relu')(block7)
    block7 = layers.Conv2D(4*filters, 3, padding='same', activation='relu')(block7)
    
    block8 = layers.UpSampling2D()(block7)
    block8 = layers.concatenate([block8, block2])
    block8 = layers.Conv2D(2*filters, 3, padding='same', activation='relu')(block8)
    block8 = layers.Conv2D(2*filters, 3, padding='same', activation='relu')(block8)
    
    block9 = layers.UpSampling2D()(block8)
    block9 = layers.concatenate([block9, block1])
    block9 = layers.Conv2D(filters, 3, padding='same', activation='relu')(block9)
    block9 = layers.Conv2D(filters, 3, padding='same', activation='relu')(block9)
    
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(block9)
    return tf.keras.Model(inputs=inputs, outputs=outputs)