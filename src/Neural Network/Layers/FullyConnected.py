import numpy as np
from Layers.Base import BaseLayer

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.weights = np.random.uniform(0,1,(output_size, input_size+1))
        self._optimizer = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return np.dot(np.hstack((self.input_tensor, np.ones((self.input_tensor.shape[0], 1)))), self.weights.T) 
    
    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    def backward(self, error_tensor):
        self.gradient_weights = np.dot(error_tensor.T, np.hstack((self.input_tensor, np.ones((self.input_tensor.shape[0], 1)))))

        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
        return np.dot(error_tensor, self.weights[:, :-1])

    
    @property
    def gradient_weights(self):
        return self._gradient_weights
    
    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value
    