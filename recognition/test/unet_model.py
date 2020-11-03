def unet_model(output_channel=4, f=4):
    inputs = tf.keras.layers.Input(shape=(256, 256, 1))

    d0 = tf.keras.layers.Conv2D(4 * f, 3, padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(inputs)
    d1 = tf.keras.layers.Conv2D(4 * f, 3, padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(d0)
    d1 = tf.keras.layers.Dropout(0.3)(d1)
    d1 = tf.keras.layers.Conv2D(4 * f, 3, padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(d1)
    op1 = d0 + d1

    op11 = tf.keras.layers.Conv2D(8 * f, 3, strides=(2, 2), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(op1)
    d2 = tf.keras.layers.Conv2D(8 * f, 3, padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(op11)
    d2 = tf.keras.layers.Dropout(0.3)(d2)
    d2 = tf.keras.layers.Conv2D(8 * f, 3, padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(d2)
    op2 = op11 + d2

    op22 = tf.keras.layers.Conv2D(16 * f, 3, strides=(2, 2), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(op2)
    d3 = tf.keras.layers.Conv2D(16 * f, 3, padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(op22)
    d3 = tf.keras.layers.Dropout(0.3)(d3)
    d3 = tf.keras.layers.Conv2D(16 * f, 3, padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(d3)
    op3 = op22 + d3

    op33 = tf.keras.layers.Conv2D(32 * f, 3, strides=(2, 2), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(op3)
    d4 = tf.keras.layers.Conv2D(32 * f, 3, padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(op33)
    d4 = tf.keras.layers.Dropout(0.3)(d4)
    d4 = tf.keras.layers.Conv2D(32 * f, 3, padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(d4)
    op4 = op33 + d4

    op44 = tf.keras.layers.Conv2D(64 * f, 3, strides=(2, 2), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(op4)
    d5 = tf.keras.layers.Conv2D(64 * f, 3, padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(op44)
    d5 = tf.keras.layers.Dropout(0.3)(d5)
    d5 = tf.keras.layers.Conv2D(64 * f, 3, padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(d5)
    op5 = op44 + d5
    # upsampling and concatente
    u4 = tf.keras.layers.UpSampling2D(size=(2, 2))(op5)
    u4 = tf.keras.layers.Conv2D(32 * f, 2, padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(u4)
    u4 = tf.keras.layers.concatenate([u4, op4])

    u3 = tf.keras.layers.Conv2D(32 * f, 3, padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(u4)
    u3 = tf.keras.layers.Conv2D(32 * f, 1, padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(u3)
    u3 = tf.keras.layers.UpSampling2D(size=(2, 2))(u3)
    u3 = tf.keras.layers.Conv2D(16 * f, 2, padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(u3)
    u3 = tf.keras.layers.concatenate([u3, op3])

    u2 = tf.keras.layers.Conv2D(16 * f, 3, padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(u3)
    u2_2 = tf.keras.layers.Conv2D(16 * f, 1, padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(u2)
    u2 = tf.keras.layers.UpSampling2D(size=(2, 2))(u2_2)
    u2 = tf.keras.layers.Conv2D(8 * f, 2, padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(u2)
    u2 = tf.keras.layers.concatenate([u2, op2])

    u1 = tf.keras.layers.Conv2D(8 * f, 3, padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(u2)
    u1_1 = tf.keras.layers.Conv2D(8 * f, 1, padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(u1)
    u1 = tf.keras.layers.UpSampling2D(size=(2, 2))(u1_1)
    u1 = tf.keras.layers.Conv2D(4 * f, 2, padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(u1)
    u1 = tf.keras.layers.concatenate([u1, op1])

    u0 = tf.keras.layers.Conv2D(8 * f, 3, padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(u1)
    u0 = tf.keras.layers.Conv2D(f, 1, padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(u0)

    sy2 = tf.keras.layers.Conv2D(f, 1, activation=tf.keras.layers.LeakyReLU(alpha=0.01))(u2_2)
    sy2 = tf.keras.layers.UpSampling2D(size=(2, 2))(sy2)
    sy1 = tf.keras.layers.Conv2D(f, 1, activation=tf.keras.layers.LeakyReLU(alpha=0.01))(u1_1)
    sy2_1 = sy2 + sy1
    sy2_1 = tf.keras.layers.UpSampling2D(size=(2, 2))(sy2_1)

    sy = u0 + sy2_1

    outputs = tf.keras.layers.Conv2D(f, 1, activation='softmax')(sy)

    return tf.keras.Model(inputs=inputs, outputs=outputs)

model = unet_model(4, f=4)
model.summary()

model.save(r'D:\PatternFlow\recognition\test\unet_model.py')

if __name__ == "__main__":
    pass
    # main()
