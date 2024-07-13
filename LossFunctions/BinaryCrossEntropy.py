from .lossfn import LossFn
import numpy as np

class BinaryCrossEntropy(LossFn):
    def __init__(self, epsilon=1e-15):
        self.epsilon = epsilon

    def loss(self, y_true, y_pred):
        # Clip y_pred to avoid log(0)
        # y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def gradient(self, y_true, y_pred):
        # Clip y_pred to avoid division by zero
        # y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        return (y_pred - y_true) / (y_pred * (1 - y_pred) * np.size(y_true))