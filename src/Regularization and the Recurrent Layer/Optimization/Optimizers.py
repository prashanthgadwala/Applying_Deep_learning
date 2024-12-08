import numpy as np

class Optimizer:
    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer
    

class Sgd(Optimizer):
    def __init__(self, learning_rate: float):
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer is not None:
            gradient_tensor += self.regularizer.calculate_gradient(weight_tensor)
        return weight_tensor - self.learning_rate * gradient_tensor


class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.learning_rate= learning_rate
        self.momentum_rate= momentum_rate
        self.velocity = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer is not None:
            gradient_tensor += self.regularizer.calculate_gradient(weight_tensor)
        self.velocity = self.momentum_rate * self.velocity - self.learning_rate * gradient_tensor
        return weight_tensor + self.velocity

class Adam(Optimizer):
    def __init__(self, learning_rate, mu, rho):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.epsilon = np.finfo(float).eps
        self.g = 0
        self.velocity = 0
        self.r = 0
        self.t = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.t += 1
        if self.regularizer is not None:
            self.g += self.regularizer.calculate_gradient(weight_tensor)
        self.g= gradient_tensor
        self.velocity = self.mu * self.velocity + (1 - self.mu) * self.g
        self.r = self.rho * self.r + (1 - self.rho) * self.g**2
        velocity_hat = self.velocity / (1 - self.mu**self.t)
        r_hat = self.r / (1 - self.rho**self.t)
        delta_w = - (self.learning_rate / (np.sqrt(r_hat) + self.epsilon)) * velocity_hat
        return weight_tensor + delta_w