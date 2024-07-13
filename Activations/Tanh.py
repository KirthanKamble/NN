from .activation import Activation
from numpy import tanh

class Tanh(Activation):
    def __init__(self):
        tanh_fn = lambda z : tanh(z)
        tanh_prime = lambda z : 1 - tanh(z)**2
        super().__init__(tanh_fn, tanh_prime)