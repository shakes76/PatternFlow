# Adjust Sigmoid

The adjust_sigmoid function performs sigmoid correction on the inputted image. The function adjusts the value of each of the pixels according to the algorithm Output = 1 / (1 + exp(gain * (cutoff - Input'))).  Input' = Input / range, where range = (max value of data type) - (min value of data type)