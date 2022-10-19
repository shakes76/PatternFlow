import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



def load_data(path, img_size):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        path, label_mode=None, color_mode = 'grayscale', image_size=img_size, batch_size=16
    )

    # normalize the data
    dataset = dataset.map(lambda x: x / 255.0)

    # unbatch the data
    dataset = dataset.unbatch()

    return dataset


def make_pair(datset1, dataset2):
    # zip the data and labels if their labels are the same (positive pairs), otherwise zip them with different labels (negative pairs)
    pos_pair1 = tf.data.Dataset.zip((datset1, datset1))
    pos_pair2 = tf.data.Dataset.zip((dataset2, dataset2))
    neg_pair1 = tf.data.Dataset.zip((datset1, dataset2))
    neg_pair2 = tf.data.Dataset.zip((dataset2, datset1))

    # label the data and labels
    pos_pair1 = pos_pair1.map(lambda x, y: (x, y, 0.0))
    pos_pair2 = pos_pair2.map(lambda x, y: (x, y, 0.0))
    neg_pair1 = neg_pair1.map(lambda x, y: (x, y, 1.0))
    neg_pair2 = neg_pair2.map(lambda x, y: (x, y, 1.0))

    return pos_pair1, pos_pair2, neg_pair1, neg_pair2


def shuffle(pos_pair1, pos_pair2, neg_pair1, neg_pair2):
    choice_dataset = tf.data.experimental.sample_from_datasets([pos_pair1, pos_pair2, neg_pair1, neg_pair2])

    return choice_dataset


def split_dataset(dataset, batch_size, train_size):
    # batch the data
    dataset = dataset.batch(batch_size)

    # split the data into train and test
    train_dataset = dataset.take(train_size)
    validation_dataset = dataset.skip(train_size)

    return train_dataset, validation_dataset


def visualize(img1, img2, labels, to_show=6, num_col=3, predictions=None, test=False):


    num_row = to_show // num_col if to_show // num_col != 0 else 1

    to_show = num_row * num_col

    # Plot the images
    fig, axes = plt.subplots(num_row, num_col, figsize=(10, 10))
    for i in range(to_show):

        # If the number of rows is 1, the axes array is one-dimensional
        if num_row == 1:
            ax = axes[i % num_col]
        else:
            ax = axes[i // num_col, i % num_col]

        ax.imshow((tf.concat([img1[i], img2[i]], axis=1).numpy()*255.0).astype("uint8"))
        ax.set_axis_off()
        if test:
            ax.set_title("True: {} | Pred: {:.5f}".format(labels[i], predictions[i][0]))
        else:
            ax.set_title("Label: {}".format(labels[i]))
    if test:
        plt.tight_layout(rect=(0, 0, 1.9, 1.9), w_pad=0.0)
    else:
        plt.tight_layout(rect=(0, 0, 1.5, 1.5))
    plt.show()


def main():
    AD_dataset = load_data('./AD_NC/train/AD', (224, 224))
    NC_dataset = load_data('./AD_NC/train/NC', (224, 224))

    pos_pair1, pos_pair2, neg_pair1, neg_pair2 = make_pair(AD_dataset, NC_dataset)

    choice_dataset = shuffle(pos_pair1, pos_pair2, neg_pair1, neg_pair2)

    train_dataset, validation_dataset = split_dataset(choice_dataset, 16, 100)

    for img1, img2, label in train_dataset.take(1):
        visualize(img1, img2, label)

        
if __name__ == '__main__':
    main()