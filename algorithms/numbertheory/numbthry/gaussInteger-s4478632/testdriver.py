
"""
A test script demonstrating the use of the module.

Student: s4478632
"""

from gaussint import *

if __name__ == "__main__":
    n1 = GaussInteger(3, 1)
    n2 = GaussInteger(1, 0)
    n3 = GaussInteger(1, 0)

    print("n1: ", n1)
    print("n2: ", n2)
    print("n3: ", n3)

    # repr
    print("repr: ", repr(n1))

    # str
    print("str: ", n1)

    # eq & ne
    print("eq: ", n2 == n3)
    print("ne: ", n2 != n3)

    # getNum
    print("getNum: ", n1.getNum())
    print("getNum type: ", type(n1.getNum()))

    # conjugate
    print("conjugate: ", n1.conjugate())

    # norm
    print("norm: ", n1.norm())

    # add
    print("add: ", n1.add(n2))
    print("add n1 + 2: ", n1.add(2))

    # __add__, __radd__, __iadd__
    print("n1 + n2: ", n1 + n2)
    print("n2 + n1: ", n2 + n1)
    n1 += 1
    print("n1 += 1: ", n1)

    # __sub__, __rsub__, __isub__
    print("-n1: ", -n1)
    print("-n1 + n2: ", -n1 + n2)
    n1 -= 1
    print("n1 -= 1: ", n1)

    # mul, __mul__, __rmul__, __imul__
    print("n1 * 1: ", n1 * 1)
    print("2 * n1: ", 2 * n1)
    n1 *= n2
    print("n1 *= n2: ", n1)

    # floordiv, __floordiv__, __ifloor__
    print("n2.floordiv(n3): ", n2.floordiv(n3))
    print("n2 // n3: ", n2 // n3)
    n1 //= n2
    print("n1 //= n2: ", n1)

    # mod, __mod__, __imod__, divmod
    print("n1.mod(n2): ", n1.mod(n2))
    print("n1 % n2: ", n1 % n2)
    n1 %= n2
    print("n1 %= n2: ", n1)
    print("n2.divmod(n3): ", n2.divmod(n3))

    # gcd, xgcd
    print("n1.gcd(n2): ", n1.gcd(n2))
    print("n1.xgcd(n2): ", n1.xgcd(n2))

    # __pow__
    print("GaussInteger(1, 1) ** GaussInteger(2, 2): ",
              GaussInteger(1, 1) ** GaussInteger(2, 2))

    # isprime
    print("GaussInteger(2, 1).isprime(): ", GaussInteger(2, 1).isprime())
    print("GaussInteger(5, 0).isprime(): ", GaussInteger(5, 0).isprime())
    print("GaussInteger(11, 0).isprime(): ", GaussInteger(11, 0).isprime())
