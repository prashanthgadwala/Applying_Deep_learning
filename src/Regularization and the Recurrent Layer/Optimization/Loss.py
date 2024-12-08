import numpy as np
from Optimization.Optimizers import Optimizer


class CrossEntropyLoss():
    
    def __init__(self):
        pass

    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor =prediction_tensor
        self.label_tensor = label_tensor
        epsilon = np.finfo(float).eps
        prediction_tensor = np.clip(prediction_tensor, epsilon, 1. - epsilon)
        loss = -np.sum(label_tensor * np.log(prediction_tensor))
        return loss

    def backward(self, label_tensor):
        epsilon = np.finfo(float).eps
        prediction_tensor = np.clip(self.prediction_tensor, epsilon, 1. - epsilon)
        return -(label_tensor / prediction_tensor) 