from .lossfn import LossFn
from numpy import size, mean, log

class CrossEntropy(LossFn):
    def loss(self, y_true, y_pred):
        return -mean(y_true * log(y_pred))

    def gradient(self, y_true, y_pred):
        return -(y_true / y_pred) / size(y_true)