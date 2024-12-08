import numpy as np

class Dropout:
    def __init__(self, probability):
        self.probability = probability  # Fraction of units to keep
        self.rate = 1 - probability  # Rate of units to drop
        self.trainable = False  # Dropout layer has no trainable parameters
        self.layer_input = None
        self.output = None
        self.binary_mask = None

    def forward(self, input, training=False):
        self.layer_input = input
        if not training:
            # During testing phase, return the input tensor unchanged
            self.output = input
        else:
            # During training, apply dropout
            self.binary_mask = (np.random.rand(*input.shape) < self.probability) / self.probability
            self.output = input * self.binary_mask
        return self.output

    def backward(self, accum_grad):
        # During backpropagation, apply the same mask to the gradient
        if self.binary_mask is not None:
            self.grad_input = accum_grad * self.binary_mask
        else:
            self.grad_input = accum_grad
        return self.grad_input

    def __repr__(self):
        return "Dropout(rate={})".format(self.rate)