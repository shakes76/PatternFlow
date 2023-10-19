import tensorflow as tf

def decode_image(filename):
    # loading the image file
    image = tf.io.read_file(filename)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.resize(image, (256, 256))
    return image

def process_image(image, seg):
    # process the image data
    image = decode_image(image)
    image = image/255.0
    
    seg = decode_image(seg)
    # tensorflow decode
    seg = tf.cast(seg == [0.0, 255.0], tf.float32)
    return image, seg