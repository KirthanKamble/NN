import numpy as np

class Reshape:
    def __init__(self, input_shape, output_shape):
        self._input_shpae = input_shape
        self._output_shpae = output_shape

    def forward(self, input_data):
        return np.reshape(input_data, self._output_shpae)
    
    def backward(self, output_grad, learning_rate=None):
        return np.reshape(output_grad, self._input_shpae)