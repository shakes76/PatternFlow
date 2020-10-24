
# Author: Naziah SIDDIQUE 
# Last update: 22/09/2019

# Barnsley's fern using numpy as it is faster than tensorflow 
import random
import matplotlib.pyplot as plt

def barnsley_arrays(points=1000):
    X = [0]
    Y = [0]

    for n in range(1, points):
        r = random.uniform(0, 100)
        if r < 1.0:
            x = 0
            y = 0.16*Y[n-1]
        elif r < 86.0:
            x = 0.85*X[n-1] + 0.04*Y[n-1]
            y = -0.04*X[n-1] + 0.85*Y[n-1]+1.6
        elif r < 93.0:
            x = 0.2*X[n-1] - 0.26*Y[n-1]
            y = 0.23*X[n-1] + 0.22*Y[n-1] + 1.6
        else:
            x = -0.15*X[n-1] + 0.28*Y[n-1]
            y = 0.26*X[n-1] + 0.24*Y[n-1] + 0.44
        X.append(x)
        Y.append(y)
        
    return X, Y


def plot_barnsley(X, Y, figsize=[9,15]):
    '''Make a plot'''
    plt.figure(figsize=figsize)
    plt.scatter(X,Y,color = 'g',marker = '.', s=0.5)
    plt.show()

def main():
    X, Y = barnsley_arrays(100000)
    plot_barnsley(X, Y)
    
if __name__ == '__main__':
    main()
