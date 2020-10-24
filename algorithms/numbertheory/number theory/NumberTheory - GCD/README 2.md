# Number Theory
Containing implementation of the Euclidean Algorithm as well as the Extended Euclidean Algorithm.
These can be imported either as a class or module.

Both algorithms require a two integer input and depending on the chosen algorithm will compute the greatest common divisor a long with the quotients needed to satisfy the equation ax + by = gcd(a,b)

## Usage

```python
from gcd import GCD
from gcdExtended_module import extended_gcd

# Class
greatComDiv = GCD(12, 4)
greatComDiv.gcd  # returns '4'

# Module
extGreatComDiv = extended_gcd(12, 4)  # returns '4, (0, 1)'
```
