import numpy as np

class TanH:
    def __init__(self):
        self.activations = None
        self.trainable = False

    def forward(self, input_tensor):
        self.activations = np.tanh(input_tensor)
        return self.activations

    def backward(self, error_tensor):
        return error_tensor * (1 - self.activations ** 2)