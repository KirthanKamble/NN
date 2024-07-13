# Neural Network Library in NumPy

This repository provides a simple yet flexible implementation of Neural Networks using NumPy. It includes various layers, activation functions, and loss functions, all encapsulated within a `Sequential` class that manages and trains the model. This project was created to gain insights into the mathematics of neural networks and deep learning, particularly the backpropagation algorithm. The implementation follows the book "Neural Networks and Deep Learning" by Michael Nielsen as a guide.

## Features

### Layers
- **Dense**: Fully connected layer.
- **Conv2D**: Convolutional layer for 2D inputs.

### Activation Functions
- **Sigmoid**
- **Tanh**
- **ReLU**
- **Softmax**

### Loss Functions
- **Mean Squared Error (MSE)**
- **Binary Cross Entropy**

### Model Management
- **Sequential**: A class to build, manage, and train the neural network model.

## Installation
Clone the repository:
```bash
[git clone https://github.com/kirthankamble/NN.git]
```

## Usage
To understand the usage of the module open any `.ipynb` notebook and get started. These notebooks demonstrate how to use the library to build, train, and evaluate neural network models

## Reference
This implementation was inspired by the book [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen.

## Future Goals
I plan to add a `.compile` and optimizers like SGD to facilitate training on larger data
