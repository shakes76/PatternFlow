import tensorflow as tf

def radon(image, theta=None, circle=True, *, preserve_range=None):
    if circle:
        shape_min = min(image.shape)
        radius = shape_min // 2
        img_shape = image.shape
        coords = tf.meshgrid(range(0, image.shape[0]), range(0, image.shape[1]))
        dist = tf.math.reduce_sum((coords - img_shape // 2) ** 2, 0)
        outside_reconstruction_circle = dist > radius ** 2
        if tf.any(image[outside_reconstruction_circle]):
            warn('Radon transform: image must be zero outside the '
                 'reconstruction circle')
        # Crop image to make it square
        slices = tuple(
            slice(int(tf.math.ceil(excess / 2)),
            (
                int(tf.math.ceil(excess / 2) + shape_min))
                if excess > 0 else
                slice(None)
            ) for excess in (img_shape - shape_min)
        )
        padded_image = image[slices]
    else:
        diagonal = tf.math.sqrt(2) * max(image.shape)
        pad = [int(tf.math.ceil(diagonal - s)) for s in image.shape]
        new_center = [(s + p) // 2 for s, p in zip(image.shape, pad)]
        old_center = [s // 2 for s in image.shape]
        pad_before = [nc - oc for oc, nc in zip(old_center, new_center)]
        pad_width = [(pb, p - pb) for pb, p in zip(pad_before, pad)]
        padded_image = tf.pad(image, pad_width, mode='constant',
            constant_values=0)