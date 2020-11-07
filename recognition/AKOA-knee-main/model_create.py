import tensorflow as tf


def create_model():
    base_model = tf.keras.applications.resnet_v2.ResNet50V2(input_shape=(224, 224, 3),
                                                            include_top=False)
    base_model.trainable = False  # Freezing base layer
    model = tf.keras.models.Sequential()
    model.add(base_model)
    model.add(tf.keras.layers.Conv2D(
        filters=8, kernel_size=1, activation='relu'))
    model.add(tf.keras.layers.Conv2D(
        filters=16, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.Conv2D(
        filters=32, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.Conv2D(
        filters=64, kernel_size=2, activation='relu'))
    model.add(tf.keras.layers.Conv2D(
        filters=128, kernel_size=1, activation='relu'))
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    return model


def plot(history):
    from matplotlib import pyplot as plt
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    #plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    #plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()
