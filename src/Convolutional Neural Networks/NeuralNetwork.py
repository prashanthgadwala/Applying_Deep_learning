import copy

class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.weights_initializer = weights_initializer
        self.bias_initializer= bias_initializer
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None

    def forward(self):
        input_tensor, self.label_tensor = self.data_layer.next()
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        loss = self.loss_layer.forward(input_tensor, self.label_tensor)
        return loss

    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        if hasattr(layer, 'trainable') and layer.trainable:
            layer.weights = self.weights_initializer.initialize(layer.weights.shape, layer.weights.shape[1],layer.weights.shape[0])
            layer.bias = self.bias_initializer.initialize(layer.bias.shape, 1, layer.bias.shape[0])
            layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)

    def train(self, iterations):
        for _ in range(iterations):
            loss = self.forward()
            self.loss.append(loss)
            self.backward()

    def test(self, input_tensor):
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return input_tensor