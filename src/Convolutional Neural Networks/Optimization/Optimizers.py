import numpy as np


class Sgd:
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        return weight_tensor - self.learning_rate * gradient_tensor


class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate= learning_rate
        self.momentum_rate= momentum_rate
        self.velocity = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.velocity = self.momentum_rate * self.velocity - self.learning_rate * gradient_tensor
        return weight_tensor + self.velocity

class Adam:
    def __init__(self, learning_rate, mu, rho):
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
        self.g= gradient_tensor
        self.velocity = self.mu * self.velocity + (1 - self.mu) * self.g
        self.r = self.rho * self.r + (1 - self.rho) * self.g**2
        velocity_hat = self.velocity / (1 - self.mu**self.t)
        r_hat = self.r / (1 - self.rho**self.t)
        delta_w = - (self.learning_rate / (np.sqrt(r_hat) + self.epsilon)) * velocity_hat
        return weight_tensor + delta_w