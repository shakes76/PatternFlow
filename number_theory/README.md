# Tensorflow Implementation of Number Theory

#### This is the readme document for COMP3710 project report.
The implementation contains modules of Euclidean algorithm (number\_theory\_gcd.py) and extended Euclidean algorithm (number\_theory\_xgcd.py) which requires tensorflow 1.15.0 to run. 


### Algorithm description
The Euclidean algorithm in number theory is an efficient method for computing the greatest common divisor (gcd) of two integers a and b. If gcd(a, b) = 1, then a and b are said to be coprime. The gcd(a,b) can be calculated by keeping divide one number by another and updating two numbers with the new remainder until the remainder becomes 0. For example, to compute gcd(48, 18), divide 48 by 18 to get a quotient of 2 and a remainder of 12. Then divide 18 by 12 to get a quotient of 1 and a remainder of 6. Then divide 12 by 6 to get a remainder of 0, which means that 6 is the gcd. 

The extended Euclidean algorithm computes gcd of two integers a and b as well as the coefficients of Bézout's identity, which are integers x and y such that ax + by = gcd(a,b). For the extended algorithm, it keeps computing a sequence of quotients and a sequence of remainders until the terminal condition (remainder equals 0) is reached.Details of Euclidean algorithm and extended Euclidean algorithm can be found here:
[Euclidean-algorithm-wikipedia](https://en.wikipedia.org/wiki/Euclidean_algorithm)
and [extended-Euclidean-algorithm-wikipedia](https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm)


### Example Usage
```python
from number_theory_gcd import gcd
from number_theory_xgcd import xgcd

gcd_s1 = gcd(56, 48) 
print(gcd_s1)  # will print: 8
xgcd_s1 = xgcd(56, 48)
print(xgcd_s1) # will print: 8, (1, -1)

```

