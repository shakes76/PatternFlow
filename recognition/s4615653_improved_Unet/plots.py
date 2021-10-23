import tensorflow as tf
import matplotlib.pyplot as plt
def plots(data, model):
    # function to plot images of inputs, ground-truth and predictions, and compared them.
    inputs_batch, truth_batch = next(iter(data.batch(3)))

    prediction = model.predict(inputs_batch)

    # inputs images
    plt.figure(figsize =(15, 15))
    for i in range(3):
        plt.subplot(3, 3, 3*i+1)
        plt.imshow(inputs_batch[i])
        plt.axis('off')

    # ground-truth images
    plt.figure(figsize = (15, 15))
    for i in range(3):
        plt.subplot(3, 3, 3*i+2)
        plt.imshow(tf.argmax(truth_batch[i]))
        plt.axis('off')

    # predict images
    plt.figure(figsize = (15, 15))
    for i in range(3):
        plt.subplot(3, 3, 3*i+2)
        plt.imshow(tf.argmax(prediction[i]))
        plt.axis('off')