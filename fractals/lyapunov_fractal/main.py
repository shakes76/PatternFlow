# -*- coding: utf-8 -*-
# Author: Antoine DELPLACE
# Last update: 19/09/2019
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import lyapunov

def plot(Efinal, vmin, vcenter, vmax, extent):
    fig = plt.figure(figsize=(10,10))
    offset = mcolors.DivergingNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    Efinal = offset(Efinal)
    plt.imshow(Efinal, extent=extent, cmap="OrRd")
    plt.colorbar()
    plt.show()


##Driver script
if __name__ == "__main__":
    # Parameters
    P0 = 0.5
    a, b = np.mgrid[2:4:0.002, 2:4:0.002]
    nb_iters = 500
    
    Efinal = lyapunov.lyapunov_exponent(P0, a, b, nb_iters)
    #print(Efinal.min(), Efinal.max())

    # Plot parameters
    vmin = -10
    vcenter = 0.
    vmax = Efinal.max()
    extent = [2,4,2,4]
    
    plot(Efinal, vmin, vcenter, vmax, extent)
