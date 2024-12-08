import numpy as np

class ReLU():
    def __init__(self):
        self.trainable = False

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return np.maximum(0, input_tensor)
    
    def backward(self, error_tensor):
        return np.multiply(error_tensor, np.where(self.input_tensor > 0, 1, 0))
    
