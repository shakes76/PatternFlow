from train import *
import tensorflow as tf
from dataset import *


def generate_prediction(test_data, model):
    prediction_batch = 20
    batched_test_data = test_data.batch(prediction_batch)
    test_image, test_mask = next(iter(batched_test_data))
    mask_prediction = model.predict(test_image)

    # Plot the original image, ground truth and result from the network.
    for i in range(prediction_batch):
        plt.figure(figsize=(10, 10))

        # Plot the test image
        plt.subplot(1, 3, 1)
        plt.imshow(test_image[i])
        plt.title("Input Image")
        plt.axis("off")

        # Plot the test mask
        plt.subplot(1, 3, 2)
        plt.imshow(test_mask[i])
        plt.title("Ground Truth Mask")
        plt.axis("off")

        # Plot the resultant mask
        plt.subplot(1, 3, 3)

        # Display 0 or 1 for classes
        prediction = tf.where(mask_prediction[i] > 0.5, 1.0, 0.0)
        plt.imshow(prediction)
        plt.title("Resultant Mask")
        plt.axis("off")

        plt.show()
    return


def main():
    loaded_model = tf.keras.models.load_model('/tmp/model')
    _, test_ds, _ = data_loader()

    generate_prediction(test_ds, loaded_model)


if __name__ == "__main__":
    main()
