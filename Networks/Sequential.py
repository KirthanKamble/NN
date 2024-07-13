class Sequential:
    def __init__(self, architecture, loss_fn):
        self.prediction = None
        self.architecture = architecture
        self.loss_fn = loss_fn

    def add(self, layer):
        self.architecture.append(layer)

    def predict(self, input_data):
        for layer in self.architecture:
            output = layer.forward(input_data)
            input_data = output

        self.prediction = output
        return self.prediction
    
    def train(self, X, y, epochs, lr):
        for epoch in range(epochs):
            error = 0
            for X_train, y_train in zip(X, y):
                # Forward pass
                y_pred = self.predict(X_train)

                # Error
                error += self.loss_fn.loss(y_true=y_train, y_pred=y_pred)

                # Backward
                grad = self.loss_fn.gradient(y_true=y_train, y_pred=y_pred)
                for layer in reversed(self.architecture):
                    grad = layer.backward(grad, lr)

            error /= len(X)  # Average the error over the entire training set
            print(f"Completed {epoch+1}/{epochs} epoch, loss = {error}")

    def loss(self, X, y):
        output = self.predict(X)
        return self.loss_fn.loss(y_true=y, y_pred=output)