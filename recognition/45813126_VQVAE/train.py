from dataset import Dataset
import matplotlib.pyplot as plt

# Load data
data = Dataset()

plt.imshow(data.train_data[0])
plt.show()

plt.imshow(data.train_labels[0,:,:,3])
plt.show()

plt.imshow(data.test_data[0])
plt.show()

plt.imshow(data.test_labels[0,:,:,3])
plt.show()

plt.imshow(data.valid_data[0])
plt.show()

plt.imshow(data.valid_labels[0,:,:,3])
plt.show()
