from layer import Layer
import numpy as np
from scipy import signal

class Convolutional2D(Layer):
    def __init__(self, input_shape, kernel_shape, num_kernels):
        input_depth, input_height, input_width = input_shape
        kernel_height, kernel_width = kernel_shape
        
        # Calculate output dimensions
        output_height = (input_height - kernel_height) + 1
        output_width = (input_width - kernel_width) + 1
        output_depth = num_kernels
        
        self.input_shape = input_shape
        self.output_shape = (output_depth, output_height, output_width)
        
        # Kernel shape should consider the depth of the input
        self.kernel_shape = (num_kernels, input_depth, kernel_height, kernel_width)
        self.bias_shape = self.output_shape
        
        # Initialize kernels and biases
        self.kernels = np.random.randn(*self.kernel_shape)
        self.biases = np.random.randn(*self.bias_shape)

    def __repr__(self):
        return (f"Convolutional2D(input_shape={self.input_shape},\n"
                f"kernel_shape={self.kernel_shape},\n"
                f"output_shape={self.output_shape}")
    
    def forward(self, input_data):
        self.input = input_data
        self.output = np.copy(self.biases)

        for i in range(self.output_shape[0]):
            for j in range(self.input_shape[0]):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
        return self.output

    def backward(self, output_grad, learning_rate):
        kernels_grad = np.zeros(self.kernel_shape)
        input_grad = np.zeros(self.input_shape)

        for i in range(self.output_shape[0]):
            for j in range(self.input_shape[0]):
                kernels_grad[i, j] = signal.correlate2d(self.input[j], output_grad[i], "valid")
                input_grad[j] = signal.correlate2d(output_grad[i], self.kernels[i, j], "full")

        self.kernels -= learning_rate * kernels_grad
        self.biases -= learning_rate * output_grad

        return input_grad