from .activation import Activation
from numpy import exp

class Sigmoid(Activation):
    def __init__(self):
        sigmoid = lambda z : 1/(1+exp(-z))
        sigmoid_prime = lambda z : (sigmoid(z)) * (1 - sigmoid(z))
        super().__init__(sigmoid, sigmoid_prime)