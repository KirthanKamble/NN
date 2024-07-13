import numpy as np

class Softmax:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input_data):
        self.input = input_data
        temp = np.exp(input_data)
        self.output =  temp / np.sum(temp)
        return self.output
    
    def backward(self, output_grad, learning_rate=None):
        n = np.size(self.output)
        y_tile = np.tile(self.output, n)
        return np.dot(y_tile * (np.identity(n) - y_tile.T), output_grad)