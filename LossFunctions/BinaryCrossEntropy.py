from .lossfn import LossFn
import numpy as np

class BinaryCrossEntropy(LossFn):
    def loss(self, y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_true))
    
    def gradient(self, y_true, y_pred):
        return ((1 - y_true / 1 - y_pred) - y_true / y_pred) / np.size(y_true)