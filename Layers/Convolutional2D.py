from layer import Layer
import numpy as np
from scipy import signal

class Convolutional2D(Layer):
    def __init__(self, input_shape, kernel_shape, num_kernels, padding=(0, 0), strides=(1, 1)):
        input_depth, input_height, input_width = input_shape
        kernel_height, kernel_width = kernel_shape
        stride_y, stride_x = strides
        padding_y, padding_x = padding
        
        # Calculate output dimensions
        output_height = ((input_height - kernel_height + 2 * padding_y) // stride_y) + 1
        output_width = ((input_width - kernel_width + 2 * padding_x) // stride_x) + 1
        output_depth = num_kernels
        
        self.input_shape = input_shape
        self.output_shape = (output_depth, output_height, output_width)

        self.strides = strides
        self.padding = padding
        
        # Kernel shape should consider the depth of the input
        self.kernel_shape = (num_kernels, input_depth, kernel_height, kernel_width)
        self.bias_shape = self.output_shape
        
        # Initialize kernels and biases
        self.kernels = np.random.randn(*self.kernel_shape)
        self.biases = np.random.randn(*self.bias_shape)

    def __repr__(self):
        return (f"Convolutional2D(input_shape={self.input_shape},\n"
                f"kernel_shape={self.kernel_shape},\n"
                f"output_shape={self.output_shape},\n"
                f"strides={self.strides}, padding={self.padding})")
    
    def forward(self, input_data):
        self.input = input_data
        self.output = np.copy(self.biases)

        for i in range(self.output_shape[0]):
            self.output[i] += 

    def backward(self, output_grad, learning_rate):
        pass

# Example usage
if __name__ == "__main__":
    
    input_shape = (3, 32, 32)  # Example: RGB image of size 32x32
    kernel_size = (3, 3)  # 3x3 kernel
    filters = 64  # 64 filters
    padding = (1, 1)  # Padding of 1 on each side
    strides = (1, 1)  # Stride of 1

    conv_layer = Convolutional2D(input_shape, kernel_size, filters, padding, strides)
    print(conv_layer)