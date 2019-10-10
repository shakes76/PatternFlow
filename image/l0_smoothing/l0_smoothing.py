import tensorflow as tf

def l0_smoothing(image):
    # Ensure that the image is a Tensor
    image = tf.convert_to_tensor(image, tf.float32)

    # Image should have 3 channels (RGB)
    assert len(image.shape) == 3, f'Image does not have shape 3. Found shape: {len(image.shape)}'

    width, height, channels = image.shape
    assert channels == 3, f'Expected image to have 3 channels but found {channels}'



    return width, height, channels

if __name__ == '__main__':
    print(l0_smoothing(tf.random.uniform([3, 3, 3])))