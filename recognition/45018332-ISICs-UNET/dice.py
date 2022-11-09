import tensorflow as tf
from tensorflow import keras as kr
from tensorflow.keras import backend as krb

def dsc(ytrue, ypred):
    #calculating dice coefficient based on formula
    ycap = krb.sum(krb.flatten(ypred)*krb.flatten(ytrue))
    yunion = krb.sum(krb.flatten(ytrue)) + krb.sum(krb.flatten(ypred))
    return ((2*ycap + 1)/(yunion + 1))

def dsc_loss(ytrue, ypred):
    #calculating dice loss as (1-dsc)
    return (1 - dsc(ytrue, ypred))