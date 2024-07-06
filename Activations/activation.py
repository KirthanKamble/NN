from numpy import multiply

class Activation:
    def __init__(self, activation, activation_prime):
        self.input = None
        self.activation = activation
        self.derv_activation = activation_prime

    def forward(self, input_data):
        self.input = input_data
        return self.activation(input_data)

    def backward(self, output_grad, learning_rate=None):
        return multiply(self.derv_activation(self.input), output_grad)