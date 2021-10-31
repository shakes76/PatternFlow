import matplotlib.pyplot as plt

epoch = range(1, 9)
val_dice = [2.545/16, 4.631/16, 4.530/16, 5.83/16, 5.58/16, 7.65/16, 7.01/16, 7.44/16]
plt.title('validation DSC')
plt.plot(epoch, val_dice)

plt.show()