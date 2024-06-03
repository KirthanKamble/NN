from numpy import maximum, where
from activation import Activation

class ReLU(Activation):
    def __init__(self):
        relu = lambda z: maximum(0, z)
        relu_prime = lambda z: where(z > 0, 1, 0)
        super().__init__(relu, relu_prime)

# Example usage of the ReLU class
if __name__ == "__main__":
    import numpy as np
    relu_layer = ReLU()

    # Forward pass
    input_data = np.array([0.1, -0.2, -0.1, 2.0, -3.0])
    output = relu_layer.forward(input_data)
    print("Forward pass output:", output)

    # Backward pass
    output_grad = np.array([1, 0.5, -0.5, 2, -2])
    input_grad = relu_layer.backward(output_grad)
    print("Backward pass output (input gradient):", input_grad)