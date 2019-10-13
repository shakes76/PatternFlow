# GaussInteger
<p>
This module implements a class representation of Gaussian integers which are of the form a+bi for integers a and b and imaginary number i. The class was ported from the gaussint.py module of the Number-Theory-Python repo of Robert-Campbell-256 (https://github.com/Robert-Campbell-256/Number-Theory-Python) to utilise tensorflow. The created class overloads various magic methods used for basic arithmetic on the Gaussian integers. It also introduces the following main methods and their dependencies (again, ported from the aforementioned module):</p>
<ul>
<li>getNum
<li>conjugate
<li>norm
<li>add
<li>mul
<li>floordiv
<li>mod
<li>divmod
<li>gcd
<li>xgcd
<li>isprime
</ul>
<p>Other methods were implemented for the prime factorisation of Gaussian integers but were ported from incomplete/possibly malfunctioning code provided in the original. Therefore these methods are exactly the same as the original and were included for completeness in the assessment piece.</p>
</br></br>
<img src="https://github.com/SoloKwansy/PatternFlow/blob/topic-algorithms/numbthry/gaussInteger-s4478632/gaussint.png"></img>
