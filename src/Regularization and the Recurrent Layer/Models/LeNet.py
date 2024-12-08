from NeuralNetwork import NeuralNetwork
from Layers import Convolution, FullyConnected, ReLU, SoftMax
from Optimization.Optimizers import Adam

def build():
    net = NeuralNetwork()
    net.add(Convolution(number_of_filters=6, filter_size=5, stride=1, padding=2, input_shape=(1, 28, 28)))
    net.add(ReLU())
    net.add(Convolution(number_of_filters=16, filter_size=5, stride=1, padding=0))
    net.add(ReLU())
    net.add(FullyConnected(output_size=120))
    net.add(ReLU())
    net.add(FullyConnected(output_size=84))
    net.add(ReLU())
    net.add(FullyConnected(output_size=10))
    net.add(SoftMax())

    optimizer = Adam(learning_rate=5e-4, regularizer=L2(weight=4e-4))
    net.compile(optimizer=optimizer)
    return net