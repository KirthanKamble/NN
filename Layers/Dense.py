from layer import Layer
from numpy import dot
from numpy.random import rand

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = randn(output_size, input_size)
        self.biases = randn(output_size, 1)
        # each row represents a neuron
        # each col represents that neurons weight for that connnection

    def forward(self, input_data):
        self.input = input_data
        return dot(self.weights, self.input) + self.biases
    
    def backward(self, output_grad, learning_rate):
        weight_grad = dot(output_grad, self.input.T)
        input_grad = dot(self.weights.T, output_grad)
        self.weights -= learning_rate * weight_grad
        self.biases -= learning_rate * output_grad
        return input_grad