class Sequential:
    def __init__(self, architecture, loss_fn):
        self.prediction = None
        self.achitecture = architecture
        self.loss_fn = loss_fn

    def add(self, layer):
        self.achitecture.append(layer)

    def predict(self, input_data):
        for layer in self.achitecture:
            output = layer.forward(input_data)
            input_data = output

        self.prediction = output
        return self.prediction
    
    def train(self, X, y, epochs, lr):
        for epoch in range(epochs):
            error = 0
            for X_train, y_train in zip(X, y):
                # forward pass
                y_pred = self.predict(X_train)

                # Error
                error += self.loss_fn.loss(y_train, y_pred)

                # Backward
                grad = self.loss_fn.gradient(y_train, y_pred)
                for layer in reversed(self.achitecture):
                    grad = layer.backward(grad, lr)

                error /= len(X_train)
                print(f"Completed {epoch+1} epoch, loss = {error}")