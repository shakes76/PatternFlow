import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
from .model import KneeClassifier

# Constants
IMAGE_DIR = './data'
BATCH_SIZE = 32
IMG_SIZE = (260, 228)
IMG_SHAPE = IMG_SIZE + (3,)
LEARNING_RATE = 0.0001

"""
Dataset creation function

For this function dataset need to be in a following folder structure
|_ data
   |_ left
      |_ left_image_1
      |_ left_image_2
      |_ ...
   |_ right
      |_ right_image_1
      |_ right_image_2
      |_ ...
"""


def create_dataset(image_dir, batch_size, img_size):
    # Training dataset
    training_dataset = image_dataset_from_directory(image_dir, shuffle=True, validation_split=0.2,
                                                    subset="training", seed=123, color_mode="rgb",
                                                    batch_size=batch_size, image_size=img_size)
    # Validation dataset
    validation_dataset = image_dataset_from_directory(image_dir, shuffle=True, validation_split=0.2,
                                                      subset="validation", seed=123, color_mode="rgb",
                                                      batch_size=batch_size, image_size=img_size)

    # Test dataset
    val_batches = tf.data.experimental.cardinality(validation_dataset)
    test_dataset = validation_dataset.take(val_batches // 5)
    validation_dataset = validation_dataset.skip(val_batches // 5)

    # buffered perfecting to load images from disk
    auto_tune = tf.data.experimental.AUTOTUNE
    training_dataset = training_dataset.prefetch(buffer_size=auto_tune)
    validation_dataset = validation_dataset.prefetch(buffer_size=auto_tune)
    test_dataset = test_dataset.prefetch(buffer_size=auto_tune)

    return training_dataset, validation_dataset, test_dataset


if __name__ == "__main__":

    # generate dataset
    training_set, validation_set, test_set = create_dataset(IMAGE_DIR, BATCH_SIZE, IMG_SIZE)

    # Class names
    class_names = training_set.class_names

    print('Number of Train batches: %d' % tf.data.experimental.cardinality(training_set))
    print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_set))
    print('Number of test batches: %d' % tf.data.experimental.cardinality(test_set))

    # Initialize the model
    knee_model = KneeClassifier(img_shape=IMG_SHAPE, no_epochs=10, train_dataset=training_set,
                                validation_dataset=validation_set, test_dataset=test_set, learning_rate=LEARNING_RATE)

    # Train the model
    history_data = knee_model.train_knee_classifier()

    # Test set evaluation
    knee_model.model_evaluation(eval_type='final')

    # View model summary
    knee_model.get_model_summary(model_type='complete')

    # Plotting the Learning curves
    acc = history_data.history['accuracy']
    val_acc = history_data.history['val_accuracy']

    loss = history_data.history['loss']
    val_loss = history_data.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

    # Compare predicted and class labels

    # Retrieve a batch of images from the test set
    image_batch, label_batch = test_set.as_numpy_iterator().next()
    predictions = knee_model.complete_model.predict_on_batch(image_batch).flatten()

    # Apply a sigmoid since our model returns logits
    predictions = tf.nn.sigmoid(predictions)
    predictions = tf.where(predictions < 0.5, 0, 1)

    print('Predictions:\n', predictions.numpy())
    print('Labels:\n', label_batch)

    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image_batch[i].astype("uint8"))
        plt.title(class_names[predictions[i]])
        plt.axis("off")


