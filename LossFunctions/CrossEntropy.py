from lossfn import LossFn
from numpy import clip, sum, log

class CrossEntropy(LossFn):
    def loss(self, y_true, y_pred):
        # Clipping y_pred to avoid log(0)
        y_pred = clip(y_pred, 1e-15, 1 - 1e-15)
        return -sum(y_true * log(y_pred)) / y_true.shape[0]

    def gradient(self, y_true, y_pred):
        # Clipping y_pred to avoid division by zero
        y_pred = clip(y_pred, 1e-15, 1 - 1e-15)
        return - (y_true / y_pred) / y_true.shape[0]