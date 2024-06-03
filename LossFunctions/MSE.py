from lossfn import LossFn
from numpy import mean, array

class MSE(LossFn):
    def loss(self, y_true, y_pred):
        return mean((y_true - y_pred) ** 2)/2

    def gradient(self, y_true, y_pred):
        return (y_pred - y_true) / y_true.size