from gcdExtended_module import extended_gcd as gcdEx
from gcd_module import gcd as GCD

'''
Example of algorithms
'''
### Euclidean Algorithm ###
'''
>>> gcd(12, 4)
4
>>> gcd(36, 27)
3
'''


### Extended Euclidean Algorithm ###
'''
>>> extended_gcd(12, 4)
4, (0, 1)
>>> extended_gcd(36, 27)
3, (1, -1)
'''


'''
Usage of algorithms
'''
a = 1296
b = 512

### Euclidean Algorithm ###
gcd = GCD(a, b)
print('---  Euclidean algorithm  ---')
print('Greatest Common Divisior of', a, 'and', b, 'is:', gcd)
print('')


### Extended Euclidean Algorithm ###
gcdE = gcdEx(a, b)
print('---  Extended Euclidean algorithm  ---')
print('Greatest Common Divisior of', a, 'and', b, 'is:', gcdE[0])
print('Related quotients are:', gcdE[1][0], 'and', gcdE[1][1])
print('Satisfying the equation:', a,'*', gcdE[1][0], '+', b, '*', gcdE[1][1], '=', gcdE[0])
