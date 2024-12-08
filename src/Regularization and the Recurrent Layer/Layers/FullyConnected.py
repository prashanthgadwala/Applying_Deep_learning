import numpy as np
from Layers.Base import BaseLayer

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.weights = np.random.uniform(0,1,(output_size, input_size+1))
        self._optimizer = None
        self.input_size = input_size    
        self.output_size = output_size
        self.bias = np.zeros((self.output_size,1))

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize((self.output_size, self.input_size), self.input_size, self.output_size)
        self.bias = bias_initializer.initialize((self.output_size,1), 1, self.output_size)
        

    def forward(self, input_tensor):
        self.input_tensor = np.hstack([input_tensor, np.ones((input_tensor.shape[0], 1))])
        output_tensor = np.dot(self.input_tensor, self.weights.T) +self.bias.T
        return output_tensor
    
    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    def backward(self, error_tensor):
        self.gradient_weights = np.dot(error_tensor.T, self.input_tensor)

        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
        return np.dot(error_tensor, self.weights[:,:-1])

    
    @property
    def gradient_weights(self):
        return self._gradient_weights
    
    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value
    