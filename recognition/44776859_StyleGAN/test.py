import numpy as np
import itertools

m1 = np.array([[1, 1], [1, 1]])
m2 = np.array([[2, 2], [2, 2]])
m3 = np.array([[3, 3], [3, 3]])
m4 = np.array([[4, 4], [4, 4]])
m5 = np.array([[5, 5], [5, 5]])
m6 = np.array([[6, 6], [6, 6]])
m7 = np.array([[7, 7], [7, 7]])
m8 = np.array([[8, 8], [8, 8]])

diff, count = np.zeros(shape=m1.shape), 0
for x in itertools.combinations((m1, m2, m3, m4, m5, m6, m7, m8), 2):
    diff = diff + np.abs(x[0] - x[1])
    count += 1
diff = diff / count
print(diff)
