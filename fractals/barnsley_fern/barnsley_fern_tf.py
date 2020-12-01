
# Barnsley's fern TENSORFLOW VERSION
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

sess = tf.Session()

# Adding points to fractal
iterations = 10000

# tensors for storing x,y coords
X = tf.Variable(np.zeros(iterations))
Y = tf.Variable(np.zeros(iterations))

# random no.s array 
randos = np.random.uniform(low=0.0, high=100.0, size=(iterations,))

# Initialise tensors 
tf.global_variables_initializer().run()

# Run step several times to generate each coord
for n in range(1,iterations):
    r = randos[n]
    prev_y = tf.gather(Y,[n-1])
    prev_x = tf.gather(X,[n-1])
    if r < 1.0:
        x = 0
        y = 0.16*prev_y
    elif r < 86.0:
        x = 0.85*prev_x + 0.04*prev_y
        y = -0.04*prev_x + 0.85*prev_y+1.6
    elif r < 93.0:
        x = 0.2*prev_x - 0.26*prev_y
        y = 0.23*prev_x + 0.22*prev_y + 1.6
    else:
        x = -0.15*prev_x + 0.28*prev_y
        y = 0.26*prev_x + 0.24*prev_y + 0.44
        
    X = tf.scatter_update(X, [n], x)
    Y = tf.scatter_update(Y, [n], y)

# Plot coordinates 
plt.figure(figsize = [6,10])
plt.scatter(X.eval(),Y.eval(),color = 'g', marker = '.', s=0.5)
plt.show()
sess.close()
