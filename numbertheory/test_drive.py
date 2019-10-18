import tensorflow as tf
import numbthy_tf as nmtf
import numbthy as nm
import numpy.random as rd

testnumber = 5
minimum = 1
maximum = 99

def gcd_function():
    print("Testing -- gcd function--")
    print("Using numpy.random generating {0} numbers".format(testnumber))
    for n in range(testnumber):   
        a = rd.randint(minimum,maximum); b = rd.randint(minimum,maximum)
        tf1 = nmtf.gcd(tf.constant(a), tf.constant(b))
        nm1 = nm.gcd(a, b)
        print("Test {0}------------------------------".format(n+1))
        print("nmtf.gcd({0}, {1}) = {2}".format(a,b,tf1.eval()))
        print("  nm.gcd({0}, {1}) = {2}".format(a,b,nm1))
        print("")

def xgcd_function():
    print("Testing -- xgcd function--")
    print("Using numpy.random generating {0} numbers".format(testnumber))
    for n in range(testnumber):   
        a = rd.randint(minimum,maximum); b = rd.randint(minimum,maximum)
        tf1 = nmtf.xgcd(tf.constant(a), tf.constant(b))
        nm1 = nm.xgcd(a, b)
        print("Test {0}------------------------------".format(n+1))
        print("nmtf.xgcd({0}, {1}) = ({2}, {3}, {4})".format(
                a,b,tf1[0].eval(),tf1[1].eval(),tf1[2].eval()))
        print("  nm.xgcd({0}, {1}) = ({2}, {3}, {4})".format(
                a,b,nm1[0],nm1[1],nm1[2]))
        print("")
        
def inverse_function():
    print("Testing -- inverse_mod function--")
    print("Since this function will raise ValueError easily from random\n"
          "generating numbers, we will use three differen pair of numbers\n"
          "to test its functionality.")
    a = 57; b = 61
    tf1 = nmtf.inverse_mod(tf.constant(a), tf.constant(b))
    nm1 = nm.inverse_mod(a, b)
    print("nmtf.inverse_mod({0}, {1}) = ({2}, {3})".format(
                a,b,tf1[0].eval(),tf1[1].eval()))
    print("  nm.inverse_mod({0}, {1}) = {2}".format(
                a,b,nm1))
    a = 23; b = 58
    tf1 = nmtf.inverse_mod(tf.constant(a), tf.constant(b))
    nm1 = nm.inverse_mod(a, b)
    print("nmtf.inverse_mod({0}, {1}) = ({2}, {3})".format(
                a,b,tf1[0].eval(),tf1[1].eval()))
    print("  nm.inverse_mod({0}, {1}) = {2}".format(
                a,b,nm1))
    print("")
    print("The last pair of numbers will make the normal inverse_mod function\n"
          "raise an ValueError. For the tensorflow version, the second parameter\n"
          "is 0")
    a = 10; b = 34
    tf1 = nmtf.inverse_mod(tf.constant(a), tf.constant(b))
    #nm1 = nm.inverse_mod(a, b)
    print("nmtf.inverse_mod({0}, {1}) = ({2}, {3})".format(
                a,b,tf1[0].eval(),tf1[1].eval()))
    #print("  nm.inverse_mod({0}, {1}) = {2}".format(
    #            a,b,nm1))

def power_function():
    print("Testing -- power_mod function --")
    print("Using numpy.random generating {0} numbers".format(testnumber))
    for n in range(testnumber):   
        a = rd.randint(minimum,maximum); b = rd.randint(minimum,maximum)
        c = rd.randint(minimum,maximum)
        tf1 = nmtf.power_mod(tf.constant(a), tf.constant(b), tf.constant(c))
        nm1 = nm.power_mod(a, b, c)
        print("Test {0}------------------------------".format(n+1))
        print("nmtf.power_mod({0}, {1}, {2}) = ({3}, {4})".format(
                a,b,c,tf1[0].eval(),tf1[1].eval()))
        print("  nm.power_mod({0}, {1}, {2}) = {3}".format(
                a,b,c,nm1))
        print("")

def isprimeF_function():
    print("Testing -- isprimeF function --")
    print("Using numpy.random generating {0} numbers".format(testnumber))
    for n in range(testnumber):
        a = rd.randint(minimum,maximum); b = rd.randint(minimum,maximum)
        tf1 = nmtf.isprimeF(tf.constant(a), tf.constant(b))
        nm1 = nm.isprimeF(a, b)
        print("Test {0}------------------------------".format(n+1))
        print("nmtf.isprimeF({0}, {1}) = {2}".format(
                a,b,tf1.eval()))
        print("  nm.isprimeF({0}, {1}) = {2}".format(
                a,b,nm1))
        print("")

def isprimeE_function():
    print("Testing -- isprimeE function --")
    print("isprimeE will occasionally fail. So, I will use some predetermine numbers")
    a = 36; b = 49
    tf1 = nmtf.isprimeE(tf.constant(a), tf.constant(b))
    nm1 = nm.isprimeE(a, b)
    print("Test 1------------------------------")
    print("nmtf.isprimeE({0}, {1}) = {2}".format(
            a,b,tf1.eval()))
    print("  nm.isprimeE({0}, {1}) = {2}".format(
            a,b,nm1))
    print("")
    a = 45; b = 87
    tf1 = nmtf.isprimeE(tf.constant(a), tf.constant(b))
    nm1 = nm.isprimeE(a, b)
    print("Test 2------------------------------")
    print("nmtf.isprimeE({0}, {1}) = {2}".format(
            a,b,tf1.eval()))
    print("  nm.isprimeE({0}, {1}) = {2}".format(
            a,b,nm1))
    print("")
    a = 12; b = 45
    tf1 = nmtf.isprimeE(tf.constant(a), tf.constant(b))
    nm1 = nm.isprimeE(a, b)
    print("Test 3------------------------------")
    print("nmtf.isprimeE({0}, {1}) = {2}".format(
            a,b,tf1.eval()))
    print("  nm.isprimeE({0}, {1}) = {2}".format(
            a,b,nm1))
    print("")
    a = 23; b = 79
    tf1 = nmtf.isprimeE(tf.constant(a), tf.constant(b))
    nm1 = nm.isprimeE(a, b)
    print("Test 4------------------------------")
    print("nmtf.isprimeE({0}, {1}) = {2}".format(
            a,b,tf1.eval()))
    print("  nm.isprimeE({0}, {1}) = {2}".format(
            a,b,nm1))
    print("")
    a = 55; b = 91
    tf1 = nmtf.isprimeE(tf.constant(a), tf.constant(b))
    nm1 = nm.isprimeE(a, b)
    print("Test 5------------------------------")
    print("nmtf.isprimeE({0}, {1}) = {2}".format(
            a,b,tf1.eval()))
    print("  nm.isprimeE({0}, {1}) = {2}".format(
            a,b,nm1))
    print("")

def is_prime_function():
    print("Testing -- is_prime function --")
    print("Using numpy.random generating {0} numbers".format(testnumber))
    for n in range(testnumber):
        a = rd.randint(minimum,maximum)
        tf1 = nmtf.is_prime(tf.constant(a))
        nm1 = nm.is_prime(a)
        print("Test {0}------------------------------".format(n+1))
        print("nmtf.is_prime({0}) = {1}".format(
                a,tf1.eval()))
        print("  nm.is_prime({0}) = {1}".format(
                a,nm1))
        print("")

def factorPR_function():
    print("Testing -- factorPR function --")
    print("Using numpy.random generating {0} numbers".format(testnumber))
    for n in range(testnumber):
        a = rd.randint(minimum,maximum)
        tf1 = nmtf.factorPR(tf.constant(a))
        nm1 = nm.factorPR(a)
        print("Test {0}------------------------------".format(n+1))
        print("nmtf.factorPR({0}) = {1}".format(
                a,tf1.eval()))
        print("  nm.factorPR({0}) = {1}".format(
                a,nm1))
        print("")
        
def factorone_function():
    print("Testing -- factorone function --")
    print("Using 5 different numbers to check")
    a = 50
    tf1 = nmtf.factorone(tf.constant(a))
    nm1 = nm.factorone(a)
    print("Test 1------------------------------")
    print("nmtf.factorone({0}) = {1}".format(
            a,tf1.eval()))
    print("  nm.factorone({0}) = {1}".format(
            a,nm1))
    print("")
    a = 77
    tf1 = nmtf.factorone(tf.constant(a))
    nm1 = nm.factorone(a)
    print("Test 2------------------------------")
    print("nmtf.factorone({0}) = {1}".format(
            a,tf1.eval()))
    print("  nm.factorone({0}) = {1}".format(
            a,nm1))
    print("")
    a = 23
    tf1 = nmtf.factorone(tf.constant(a))
    nm1 = nm.factorone(a)
    print("Test 3------------------------------")
    print("nmtf.factorone({0}) = {1}".format(
            a,tf1.eval()))
    print("  nm.factorone({0}) = {1}".format(
            a,nm1))
    print("")
    a = 12
    tf1 = nmtf.factorone(tf.constant(a))
    nm1 = nm.factorone(a)
    print("Test 4------------------------------")
    print("nmtf.factorone({0}) = {1}".format(
            a,tf1.eval()))
    print("  nm.factorone({0}) = {1}".format(
            a,nm1))
    print("")
    a = 39
    tf1 = nmtf.factorone(tf.constant(a))
    nm1 = nm.factorone(a)
    print("Test 5------------------------------")
    print("nmtf.factorone({0}) = {1}".format(
            a,tf1.eval()))
    print("  nm.factorone({0}) = {1}".format(
            a,nm1))
    print("")

with tf.compat.v1.Session() as sess:
    print("nmtf stands for numbthy_tf module (my own module)");
    print("nm stands for numbthy module (original module)");
    gcd_function()
    print("")
    xgcd_function()
    print("")
    inverse_function()
    print("")
    power_function()
    print("")
    isprimeF_function()
    print("")
    isprimeE_function()
    print("")
    is_prime_function()
    print("")
    factorPR_function()
    print("")
    factorone_function()
