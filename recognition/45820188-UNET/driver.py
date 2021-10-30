from model import build_model
import tensorflow as tf
import matplotlib.pyplot as plt

batch_size = 16
depth = 16
epochs = 5
n = 192
m = 256

def process_images(path, segmentation):
    return tf.keras.preprocessing.image_dataset_from_directory(
        directory = path, 
        labels=None,
        label_mode = 'binary',
        batch_size = batch_size,
        validation_split = 0.1,
        subset=segmentation,
        image_size = (n,m),
        color_mode = 'grayscale',
        shuffle = True,
        seed = 45820188
    )

model = build_model(input_shape=(n,m,1), depth=depth)

# Full Dataset
X_train_ds = process_images("C:\ISIC Dataset\Full Set\ISIC2018_Task1-2_Training_Input_x2", "training")
y_train_ds = process_images("C:\ISIC Dataset\Full Set\ISIC2018_Task1_Training_GroundTruth_x2", "training")

X_test_ds = process_images("C:\ISIC Dataset\Full Set\ISIC2018_Task1-2_Training_Input_x2", "validation")
y_test_ds = process_images("C:\ISIC Dataset\Full Set\ISIC2018_Task1_Training_GroundTruth_x2", "validation")


# Smaller subset of data
# X_train_ds = process_images("C:\ISIC Dataset\Smaller\Train", "training")
# y_train_ds = process_images("C:\ISIC Dataset\Smaller\Seg", "training")

# X_test_ds = process_images("C:\ISIC Dataset\Smaller\Train", "validation")
# y_test_ds = process_images("C:\ISIC Dataset\Smaller\Seg", "validation")


X_train = tf.concat([x for x in X_train_ds], axis=0)
y_train = tf.concat([x for x in y_train_ds], axis=0)
X_test = tf.concat([x for x in X_test_ds], axis=0)
y_test = tf.concat([x for x in y_test_ds], axis=0)

# Check that data has been processed correctly
plt.figure(figsize=(10, 10))
for images in y_test_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.axis("off")
plt.show()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(X_test, y_test))

# Final Output of images
# Compares original with the expected and actual output
prediction = model.predict(X_test)
plt.figure(figsize=(10, 10))
n = 4
for i in range(n):
    plt.subplot(n, 3, i*3+1)
    plt.imshow(X_test[i])
    plt.axis('off')
    plt.title("Original", size=11)
    plt.subplot(n, 3, i*3+2)
    plt.imshow(prediction[i])
    plt.axis('off')
    plt.title("Prediction", size=11)
    plt.subplot(n, 3, i*3+3)
    plt.imshow(y_test[i])
    plt.axis('off')
    plt.title("Expected", size=11)
plt.show()