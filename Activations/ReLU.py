from numpy import maximum, where
from activation import Activation

class ReLU(Activation):
    def __init__(self):
        relu = lambda z: maximum(0, z)
        relu_prime = lambda z: where(z > 0, 1, 0)
        super().__init__(relu, relu_prime)