import matplotlib.pyplot as plt
from adjust_log_transform import adjust_log


img = plt.imread("./11.jpeg")
img2 = adjust_log(img)
img3 = adjust_log(img, inv=True)
plt.figure(figsize=(10, 3))
plt.subplot(1, 3, 1)
plt.imshow(img)
plt.axis('off')
plt.title('Origin')

plt.subplot(1, 3, 2)
plt.imshow(img2)
plt.axis('off')
plt.title('image, gain=1, inv=False')

plt.subplot(1, 3, 3)
plt.imshow(img3)
plt.axis('off')
plt.title('image, gain=1, inv=True')

plt.savefig('output.png')
