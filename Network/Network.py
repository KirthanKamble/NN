class Network:
    def __init__(self, architecture, loss_fn):
        self.prediction = None
        self.achitecture = architecture
        self.loss_fn = loss_fn

    def predict(self, input_data):
        for layer in self.achitecture:
            output = layer.forward(input_data)
            input_data = output

        self.prediction = output
        return self.prediction
    
    def train(self, X, y, epochs, lr):
        for epoch in range(epochs):
            