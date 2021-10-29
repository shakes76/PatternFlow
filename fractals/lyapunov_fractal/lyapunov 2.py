# -*- coding: utf-8 -*-
# Author: Antoine DELPLACE
# Last update: 19/09/2019
import tensorflow as tf
import numpy as np

##Lyapunov fractal
def lyapunov_exponent(P0, a, b, nb_iters):
    sess = tf.InteractiveSession()

    atf = tf.Variable(a.astype(np.float32))
    btf = tf.Variable(b.astype(np.float32))
    Pn = tf.Variable(tf.fill(tf.shape(a), P0))
    n = tf.Variable([0.])
    E = tf.Variable(tf.zeros_like(Pn, tf.float32))

    tf.global_variables_initializer().run() #init variables

    # The chosen sequence is: AB
    Pn_ = atf*Pn*(1-Pn) #A
    Pnp_ = btf*(1-2*Pn_)
    Pn_2 = btf*Pn_*(1-Pn_) #B
    Pnp_2 = atf*(1-2*Pn_2)

    # Computing Lyapunov exponent
    n_ = n+2
    E_ = 1/n_*(E*n+tf.log(tf.abs(Pnp_))+tf.log(tf.abs(Pnp_2)))

    step = tf.group(n.assign(n_),
                    E.assign(E_),
                    Pn.assign(Pn_2))

    for i in range(nb_iters):
        step.run()

    Efinal = E.eval()

    sess.close()

    return Efinal
