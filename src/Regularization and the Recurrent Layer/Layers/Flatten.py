import numpy as np

class Flatten:
    def __init__(self):
        self.input_shape = None
        self.trainable = False

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape
        return np.reshape(input_tensor, (input_tensor.shape[0], -1))

    
    def backward(self, error_tensor):
        return np.reshape(error_tensor, self.input_shape)

