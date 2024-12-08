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
        self.phase = 'train'
        for _ in range(iterations):
            regularization_loss = 0
            loss = self.forward()
            for layer in self.layers:
                if hasattr(layer, 'regularizer') and layer.regularizer is not None:
                    regularization_loss += layer.regularizer.norm(layer.weights)
            
            total_loss = loss + regularization_loss 
            self.loss.append(total_loss)
            self.backward()

    def test(self, input_tensor):
        self.phase = 'test'
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return input_tensor
    
    @property
    def phase(self):
        return self.data_layer.phase
    
    @phase.setter
    def phase(self, value):
        for layer in self.layers:
            if hasattr(layer, 'phase'):
                layer.phase = value
    
