import numpy as np
from scipy.signal import convolve, correlate
import copy

class Conv:
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        self.stride_shape = (stride_shape[0], stride_shape[0]) if isinstance(stride_shape, list) else (stride_shape if isinstance(stride_shape, tuple) else (stride_shape, stride_shape))
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.trainable = True
        self.weights = np.random.uniform(0, 1, (num_kernels,) + convolution_shape)
        self.bias = np.random.uniform(0, 1, (num_kernels,))
        self.optimizer = None

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias
    
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        if optimizer is not None:
            self._optimizer = [copy.deepcopy(optimizer), copy.deepcopy(optimizer)]
        else:
            self._optimizer = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor

        if input_tensor.ndim == 3:
            output_shape = ((input_tensor.shape[2] + self.stride_shape[0] - 1) // self.stride_shape[0],)
        else:
            output_height = (input_tensor.shape[2] + self.stride_shape[0] - 1) // self.stride_shape[0] 
            output_width = (input_tensor.shape[3] + self.stride_shape[1] - 1) // self.stride_shape[1]  
            output_shape = (output_height, output_width)

        output_tensor = np.zeros((input_tensor.shape[0], self.num_kernels) + output_shape)

        for batch in range(input_tensor.shape[0]):
            for kernel in range(self.num_kernels):
                for channel in range(self.convolution_shape[0]):
                    correlation = correlate(input_tensor[batch, channel], self.weights[kernel, channel], 'same')
                    if correlation.ndim == 1:
                        correlation = correlation[::self.stride_shape[0]]
                    else:
                        correlation = correlation[::self.stride_shape[0], ::self.stride_shape[1]]
                    if correlation.shape != output_tensor[batch, kernel].shape:
                        correlation = correlation[:output_tensor[batch, kernel].shape[0]]
                    output_tensor[batch, kernel] += correlation

                output_tensor[batch, kernel] += self.bias[kernel]

        return output_tensor
    
    def backward(self, error_tensor):
        self.error_tensor = error_tensor
        batch_size = self.input_tensor.shape[0]
        output_tensor = np.zeros_like(self.input_tensor)
        self._gradient_weights = np.zeros_like(self.weights)

        if self.input_tensor.ndim == 4:
            self._gradient_bias = np.sum(error_tensor, axis=(0, 2, 3))
        else:
            self._gradient_bias = np.sum(error_tensor, axis=(0, 2))

        for batch in range(batch_size):
            for kernel in range(self.num_kernels):
                for channel in range(self.convolution_shape[0]):
                    if self.input_tensor.ndim == 3:
                        conv_result = correlate(self.input_tensor[batch, channel], error_tensor[batch, kernel], mode='valid')
                        conv_result = conv_result[::self.stride_shape[0]]
                        if conv_result.shape != self._gradient_weights[kernel, channel].shape:
                            conv_result = conv_result[:self._gradient_weights[kernel, channel].shape[0]]
                        self._gradient_weights[kernel, channel] += conv_result

                        grad_input = convolve(self.weights[kernel, channel], error_tensor[batch, kernel], mode='full')
                        grad_input = grad_input[::self.stride_shape[0]]
                        if grad_input.shape != output_tensor[batch, channel].shape:
                            grad_input = grad_input[:output_tensor[batch, channel].shape[0]]
                        output_tensor[batch, channel] += grad_input

                    elif self.input_tensor.ndim == 4:
                        conv_result = correlate(self.input_tensor[batch, channel], error_tensor[batch, kernel], mode='valid')
                        conv_result = conv_result[::self.stride_shape[0], ::self.stride_shape[1]]
                        if conv_result.shape != self._gradient_weights[kernel, channel].shape:
                            conv_result = conv_result[:self._gradient_weights[kernel, channel].shape[0]]
                        self._gradient_weights[kernel, channel] += conv_result

                        grad_input = convolve(self.weights[kernel, channel], error_tensor[batch, kernel], mode='full')
                        grad_input = grad_input[::self.stride_shape[0], ::self.stride_shape[1]]
                        if grad_input.shape != output_tensor[batch, channel].shape:
                            grad_input = grad_input[:output_tensor[batch, channel].shape[0]]
                        output_tensor[batch, channel] += grad_input
                output_tensor[batch, channel] += self.bias[kernel]

        if self.optimizer is not None:
            self.weights = self.optimizer[0].calculate_update(self.weights, self._gradient_weights)
            self.bias = self.optimizer[1].calculate_update(self.bias, self._gradient_bias)

        return output_tensor

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.weights.shape, np.prod(self.convolution_shape), np.prod(self.convolution_shape[1:]) * self.num_kernels)
        self.bias = bias_initializer.initialize(self.bias.shape, np.prod(self.convolution_shape), np.prod(self.convolution_shape[1:]) * self.num_kernels)
