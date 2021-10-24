import tensorflow as tf
import matplotlib.pyplot as plt
def plots(data, model):
    # function to plot images of inputs, ground-truth and predictions, and compared them.
    inputs_batch, truth_batch = next(iter(data.batch(3)))

    prediction = model.predict(inputs_batch)

    plt.figure(figsize =(15, 15))
    for i in range(3):
        # inputs images
        plt.subplot(3, 3, 3*i+1)
        plt.imshow(inputs_batch[i])
        plt.axis('off')

        # ground-truth images
        plt.subplot(3, 3, 3*i+2)
        plt.imshow(tf.argmax(truth_batch[i],axis=-1),camp="gray")
        plt.axis('off')

        # predict images
        plt.subplot(3, 3, 3*i+3)
        plt.imshow(tf.argmax(prediction[i],axis=-1),camp="gray")
        plt.axis('off')

    plt.show()