import numpy as np
from Layers.Helpers import compute_bn_gradients

class BatchNormalization:
    def __init__(self, channels):
        self.channels = channels
        self.trainable = True
        self.gamma = None
        self.beta = None
        self.initialize()
        self.epsilon = 1e-10
        self.mean = None
        self.var = None
        self.moving_mean = None
        self.moving_var = None
        self.optimizers = None  # Placeholder for optimizers

    def initialize(self):
        self.gamma = np.ones((self.channels,))
        self.beta = np.zeros((self.channels,))

    def forward(self, input_tensor, training=False):
        reshaped_input = self.reformat(input_tensor, to_vector=True)
        
        if training:
            self.mean = np.mean(reshaped_input, axis=0)
            self.var = np.var(reshaped_input, axis=0)
            if self.moving_mean is None:
                self.moving_mean = self.mean
                self.moving_var = self.var
            else:
                self.moving_mean = 0.9 * self.moving_mean + 0.1 * self.mean
                self.moving_var = 0.9 * self.moving_var + 0.1 * self.var
            normalized_input = (reshaped_input - self.mean) / np.sqrt(self.var + self.epsilon)
        else:
            normalized_input = (reshaped_input - self.moving_mean) / np.sqrt(self.moving_var + self.epsilon)
        
        output = self.gamma * normalized_input + self.beta
        return self.reformat(output, to_vector=False)

    def backward(self, error_tensor):
        reshaped_error = self.reformat(error_tensor, to_vector=True)
        reshaped_input = self.reformat(self.input_tensor, to_vector=True)
        
        # Use provided helper function for gradient computation
        grad_input = compute_bn_gradients(reshaped_error, reshaped_input, self.gamma, self.mean, self.var)
        
        # Update gamma and beta if optimizers are defined
        if self.optimizers is not None:
            self.gamma -= self.optimizers['gamma'].update(self.gamma, grad_input)
            self.beta -= self.optimizers['beta'].update(self.beta, grad_input)
        
        return self.reformat(grad_input, to_vector=False)

    def reformat(self, tensor, to_vector):
        if to_vector:
            if tensor.ndim == 4:  # Image-like to vector-like
                return tensor.reshape(tensor.shape[0], -1)
            return tensor  # Already vector-like
        else:
            if tensor.ndim == 2 and self.input_shape:  # Vector-like to image-like
                return tensor.reshape(self.input_shape)
            return tensor  # Already image-like or shape not known
