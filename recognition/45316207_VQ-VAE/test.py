import dataset
import utils
import numpy as np
import matplotlib.pyplot as plt

(train_data, test_data, validate_data, data_variance) = dataset.load_dataset(max_images=20)


num_examples_to_generate = 16
idx = np.random.choice(len(train_data), num_examples_to_generate)

print(f"===========================================")

# Method 1
# # print(f"### {idx} ###")
# print(f"### {np.shape(train_data)} ###")
# test_images = train_data[idx]
# for index in range(0, 10):
#     utils.show_subplot(train_data[index], train_data[index])

# Method 2
# fig = plt.figure(figsize=(16, 16))

# for i in range(0, 16):
#     plt.subplot(4, 4, i + 1)
#     plt.imshow(train_data[i, :, :, 0], cmap='gray')
#     # plt.imshow(reconstructions_test[i, :, :, 0])
#     plt.axis('off')

plt.savefig('test.png')
plt.show()
# plt.close()