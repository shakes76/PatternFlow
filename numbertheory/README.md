# Number Theoretic Functions implemented in Tensorflow
##### COMP3170 Pattern Recognition and Analysis (Final Project)

numbthy\_tf.py module requires tensorflow 2.0 to run

numbthy\_tf contains the following functions:

gcd(a,b), xgcd(a,b), power\_mod(b,e,n), inverse\_mod(b,n), is\_prime(n), isprimeF(n,b), 
isprimeE(n,b), factorone(n) and factorPR(n) 

(Exact descriptions are in the numbthy\_tf.py itself)

The original algorithmns are created by Robert-Campbell-256. The files can be found here:
[Robert-Campbell-256/Number-Theory-Python/numbthy.py]
(https://github.com/Robert-Campbell-256/Number-Theory-Python/blob/master/numbthy.py)
Robert-Campbell-256 uses numpy, functools and math module to construct the number theoretics functions. 
Whereas, this module uses the same algorithmns, but in tensorflow version of them. So, to run
this module, users are required to import tensorflow and run a session in it. 

E.g.
```python
import tensorflow as tf
import numbthy_tf as nm

sess = tf.InteractiveSession()

tf.global_variable_initializer().run()

integer_1 = tf.constant(123)
integer_2 = tf.constant(24)
result = nm.gcd(integer_1, integer_2)
print(result.eval())    #it will print 3
```
