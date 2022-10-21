import numpy as np
import matplotlib.pyplot as plt


def show_original_vs_reconstructed(original, reconstructed):
    plt.subplot(1, 2, 1)
    #plt.imshow(original.squeeze() + 0.5)
    plt.imshow(original/255., cmap='gray' )
    print("original.shape", original.shape)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    #plt.imshow(reconstructed.squeeze() + 0.5)
    plt.imshow(reconstructed, cmap='gray' )
    plt.title("Reconstructed")
    plt.axis("off")
    plt.show()

def show_history(history):
    total_loss = history['total_loss']
    vq_loss = history['vq_loss']

    epochs = range(1, len(total_loss) + 1)

    plt.plot(epochs, total_loss, 'bo', label='Training loss')
    plt.plot(epochs, vq_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()