import numpy as np

class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_state = np.zeros(hidden_size)
        self._memorize = False
        # Initialize weights and biases
        self.weights = np.random.randn(hidden_size, input_size + hidden_size)
        self.biases = np.random.randn(hidden_size)
        self.optimizer = None
        self.inputs = []
        self.outputs = []
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_biases = np.zeros_like(self.biases)

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, value):
        self._memorize = value

    def forward(self, input_tensor):
        batch_size, time_steps, _ = input_tensor.shape
        self.outputs = []
        self.inputs = []
        if not self._memorize:
            self.hidden_state = np.zeros((batch_size, self.hidden_size))
        for t in range(time_steps):
            combined_input = np.hstack((input_tensor[:, t, :], self.hidden_state))
            self.hidden_state = np.tanh(np.dot(combined_input, self.weights.T) + self.biases)
            self.outputs.append(self.hidden_state)
            self.inputs.append(combined_input)
        return np.array(self.outputs).transpose(1, 0, 2)

    def backward(self, error_tensor):
        # Simplified backward pass
        # Actual implementation would involve calculating gradients for weights, biases, and inputs
        dW = np.zeros_like(self.weights)
        dB = np.zeros_like(self.biases)
        dX = np.zeros((error_tensor.shape[0], error_tensor.shape[1], self.input_size))
        for t in reversed(range(error_tensor.shape[1])):
            # Gradient calculation logic here
            pass
        self.grad_weights = dW
        self.grad_biases = dB
        return dX

    @property
    def gradient_weights(self):
        return self.grad_weights, self.grad_biases

    @gradient_weights.setter
    def gradient_weights(self, value):
        self.grad_weights, self.grad_biases = value

    def add_optimizer(self, optimizer):
        self.optimizer = optimizer

    def calculate_regularization_loss(self):
        # Simplified regularization loss calculation
        # Actual implementation depends on the regularization method
        return 0.0

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer(self.weights.shape)
        self.biases = bias_initializer(self.biases.shape)