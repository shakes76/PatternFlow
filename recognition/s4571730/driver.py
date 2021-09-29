import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
from model import fourier_encode

# Constants
IMAGE_DIR = 's4571730/AKOA_Analysis'
BATCH_SIZE = 32
IMG_SIZE = (260, 228)
IMG_SHAPE = (260, 228, 3)
LR = 0.0001
TEST_PORTION = 5
SHUFFLE_RATE = 512
AUTO_TUNE = tf.data.experimental.AUTOTUNE

"""
Create train, validate and test dataset
Images have to be in left and right folders 

    AKOA_Analysis/
    ...left/
    ......left_image_1.jpg
    ......left_image_2.jpg
    ...right/
    ......right_image_1.jpg
    ......right_image_2.jpg

"""
def create_dataset(image_dir, batch_size, img_size):
    # Training dataset, shuffle is True by default
    training_dataset = image_dataset_from_directory(image_dir, validation_split=0.2, color_mode='grayscale',
                                                    subset="training", seed=46, label_mode='int',
                                                    batch_size=batch_size, image_size=img_size)
    # Validation dataset
    validation_dataset = image_dataset_from_directory(image_dir, validation_split=0.2, color_mode='grayscale',
                                                      subset="validation", seed=46, label_mode='int',
                                                      batch_size=batch_size, image_size=img_size)

    # Test dataset, taking 1/5 of the validation set
    val_batches = tf.data.experimental.cardinality(validation_dataset)
    test_dataset = validation_dataset.take(val_batches // TEST_PORTION)
    validation_dataset = validation_dataset.skip(val_batches // TEST_PORTION)

    # normalize and prefetch images for faster training
    training_dataset = training_dataset.map(process).prefetch(AUTO_TUNE)
    validation_dataset = validation_dataset.map(process).prefetch(AUTO_TUNE)
    test_dataset = test_dataset.map(process).prefetch(AUTO_TUNE)

    return training_dataset, validation_dataset, test_dataset

"""
Normalize image to range [0,1]
"""
def process(image,label):
    image = tf.cast(image / 255. ,tf.float32)
    return image,label

def runner_code():
    pass

if __name__ == "__main__":

    # generate dataset
    training_set, validation_set, test_set = create_dataset(IMAGE_DIR, BATCH_SIZE, IMG_SIZE)

    for image, label in training_set:
        # train_image = image[0]
        b, *axis, _ = image.shape
        axis_pos = list(map(lambda size: tf.linspace(-1.0, 1.0, num=size), axis))
        pos = tf.stack(tf.meshgrid(*axis_pos, indexing="ij"), axis=-1)
        encode = fourier_encode(pos, 4, 10)
        print(encode.shape)
        break

    # print('Number of Train batches: %d' % tf.data.experimental.cardinality(training_set))
    # print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_set))
    # print('Number of test batches: %d' % tf.data.experimental.cardinality(test_set))

    # # Initialize the model
    # knee_model = KneeClassifier(img_shape=IMG_SHAPE, no_epochs=10, train_dataset=training_set,
    #                             validation_dataset=validation_set, test_dataset=test_set, learning_rate=LEARNING_RATE)

    # # Train the model
    # history_data = knee_model.train_knee_classifier()

    # # Test set evaluation
    # knee_model.model_evaluation(eval_type='final')

    # # View model summary
    # knee_model.get_model_summary(model_type='complete')

    # # Plotting the Learning curves
    # acc = history_data.history['accuracy']
    # val_acc = history_data.history['val_accuracy']

    # loss = history_data.history['loss']
    # val_loss = history_data.history['val_loss']

    # plt.figure(figsize=(8, 8))
    # plt.subplot(2, 1, 1)
    # plt.plot(acc, label='Training Accuracy')
    # plt.plot(val_acc, label='Validation Accuracy')
    # plt.legend(loc='lower right')
    # plt.ylabel('Accuracy')
    # plt.ylim([min(plt.ylim()), 1])
    # plt.title('Training and Validation Accuracy')

    # plt.subplot(2, 1, 2)
    # plt.plot(loss, label='Training Loss')
    # plt.plot(val_loss, label='Validation Loss')
    # plt.legend(loc='upper right')
    # plt.ylabel('Cross Entropy')
    # plt.ylim([0, 1.0])
    # plt.title('Training and Validation Loss')
    # plt.xlabel('epoch')
    # plt.show()

    # # Compare predicted and class labels

    # # Retrieve a batch of images from the test set
    # image_batch, label_batch = test_set.as_numpy_iterator().next()
    # predictions = knee_model.complete_model.predict_on_batch(image_batch).flatten()

    # # Apply a sigmoid since our model returns logits
    # predictions = tf.nn.sigmoid(predictions)
    # predictions = tf.where(predictions < 0.5, 0, 1)

    # print('Predictions:\n', predictions.numpy())
    # print('Labels:\n', label_batch)

    # plt.figure(figsize=(10, 10))
    # for i in range(9):
    #     ax = plt.subplot(3, 3, i + 1)
    #     plt.imshow(image_batch[i].astype("uint8"))
    #     plt.title(class_names[predictions[i]])
    #     plt.axis("off")


